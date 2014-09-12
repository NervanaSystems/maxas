package MaxAs;

use strict;
use Data::Dumper;
use MaxAsGrammar;

require 5.10.0;

# For debugging the scheduler and parser
my $DEBUG = 0;

# these ops need to be converted from absolute addresses to relative in the sass output by cuobjdump
my %relOffset  = map { $_ => 1 } qw(BRA SSY CAL PBK PCNT);

# these ops use absolute addresses
my %absOffset  = map { $_ => 1 } qw(JCAL);

# These instructions use r0 but do not write to r0
my %noDest     = map { $_ => 1 } qw(ST STG STS STL RED);

# Map register slots to reuse control codes
my %reuseSlots = (r8 => 1, r20 => 2, r39 => 4);

# Preprocess and Assemble a source file
sub Assemble
{
    my ($file, $doReuse) = @_;

    $file = Preprocess($file, 1);

    # initialize cubin counts
    my $regCnt = 0;
    my $barCnt = 0;
    my $insCnt = 0;

    my ($lineNum, @instructs, %labels, $ctrl, @branches, %reuse);

    # initialize the first control instruction
    push @instructs, $ctrl = {};

    foreach my $line (split "\n", $file)
    {
        # keep track of line nums in the physical file
        $lineNum++;

        next unless preProcessLine($line);

        # match an instruction
        if (my $inst = processAsmLine($line, $lineNum))
        {
            # count the instructions for updating the cubin
            $insCnt += 1 if $inst->{op} ne 'EXIT';

            # track branches/jumps/calls/etc for label remapping
            push @branches, @instructs+0 if exists($relOffset{$inst->{op}}) || exists($absOffset{$inst->{op}});

            push @{$ctrl->{ctrl}}, $inst->{ctrl};

            # add the op name and full instruction text
            push @instructs, $inst;

            # add a 4th control instruction for every 3 instructions
            push @instructs, $ctrl = {} if ((@instructs & 3) == 0);
        }
        # match a label
        elsif ($line =~ m'^([a-zA-Z]\w*):')
        {
            # map the label name to the index of the instruction about to be inserted
            $labels{$1} = @instructs+0;
        }
        else
        {
            die "badly formed line at $lineNum: $line\n";
        }
    }
    # add the final BRA op and align the number of instructions to a multiple of 8
    push @{$ctrl->{ctrl}}, 0x007ff;
    push @instructs, { op => 'BRA', inst => 'BRA 0xfffff8;' };
    while (@instructs & 7)
    {
        push @instructs, $ctrl = {} if ((@instructs & 3) == 0);
        push @{$ctrl->{ctrl}}, 0x007e0;
        push @instructs, { op => 'NOP', inst => 'NOP;' };
    }

    # remap labels
    foreach my $i (@branches)
    {
        if ($instructs[$i]{inst} !~ m'(\w+);$' || !exists $labels{$1})
            { die "instruction has invalid label: $instructs[$i]{inst}"; }

        if (exists $relOffset{$instructs[$i]{op}})
            { $instructs[$i]{inst} =~ s/(\w+);$/sprintf '0x%06x;', (($labels{$1} - $i - 1) * 8) & 0xffffff/e; }
        else
            { $instructs[$i]{inst} =~ s/(\w+);$/sprintf '0x%06x;', ($labels{$1} * 8) & 0xffffff/e; }
    }

    # assemble the instructions to op codes
    foreach my $i (0 .. $#instructs)
    {
        unless ($i & 3)
        {
            $ctrl = $instructs[$i];
            next;
        }
        my ($op, $inst) = @{$instructs[$i]}{qw(op inst)};

        my $match = 0;
        foreach my $gram (@{$grammar{$op}})
        {
            # Apply the rule pattern
            next unless $inst =~ $gram->{rule};

            # get any vector registers for r0
            my @r0 = exists($+{r0}) ? getVecRegisters($+{r0}, $+{type}) : ();

            # update the register count
            foreach my $r (qw(r0 r8 r20 r39))
            {
                if (exists($+{$r}) && $+{$r} ne 'RZ')
                {
                    my $val = substr $+{$r}, 1;

                    my $regInc = $r eq 'r0' ? scalar(@r0) : 1;

                    # smart enough to count vector registers for memory instructions.
                    $regCnt = $val + $regInc if $val + $regInc > $regCnt;
                }
            }

            # update the barrier resource count
            if ($op eq 'BAR')
            {
                if (exists $+{i8w4})
                {
                    $barCnt = $+{i8w4}+1 if $+{i8w4}+1 > $barCnt;
                }
                # if a barrier value is a register, assume the maximum
                elsif (exists $+{r8})
                {
                    $barCnt = 16;
                }
            }

            # We apply the reuse logic here since it's much easier than dealing with the disassembled code.
            # There are 2 reuse slots per register slot
            # The reuse hash points to most recent instruction index where register was last used in this slot
            if ($doReuse)
            {
                # For writes to a register, clear any reuse opportunity
                if (@r0 && $r0[0] ne 'RZ' && !exists $noDest{$op})
                {
                    foreach my $slot (keys %reuseSlots)
                    {
                        if (my $reuse = $reuse{$slot})
                        {
                            # if writing with a vector op, clear all linked registers
                            delete $reuse->{$_} foreach @r0;
                        }
                    }
                }

                # only track register reuse for instruction types this works with
                if ($gram->{type}{reuse})
                {
                    foreach my $slot (keys %reuseSlots)
                    {
                        next unless exists $+{$slot};

                        my $r = $+{$slot};
                        next if $r eq 'RZ';
                        next if $r eq $+{r0}; # dont reuse if we're writing this reg in the same instruction

                        my $reuse = $reuse{$slot} ||= {};

                        # if this register was previously marked for potential reuse
                        if (my $p = $reuse->{$r})
                        {
                            # flag the previous instruction
                            $instructs[$p]{reuse}{$slot}++;
                        }
                        # list full, delete the oldest
                        elsif (keys %$reuse > 2)
                        {
                            my $oldest = (sort {$reuse->{$a} <=> $reuse->{$b}} keys %$reuse)[0];
                            delete $reuse->{$oldest};
                        }
                        # mark the new instruction for potential reuse
                        $reuse->{$r} = $i;
                    }
                }
            }

            # Generate the op code.  Note that we also get a reuse code back.
            # For some instruction types this is overloaded for a different purpose.
            # We want to keep that value.  Otherwise we'll overwrite with the above calculated value (on the final pass).
            my ($code, $reuse) = genCode($op, $gram);

            $instructs[$i]{code} = $code;
            push @{$ctrl->{reuse}}, $reuse;

            $match = 1;
            last;
        }
        unless ($match)
        {
            print "$_->{rule}\n\n" foreach @{$grammar{$op}};
            die "Unable to encode instruction: $inst\n";
        }
    }

    # Another pass to update reuse flags for control instructions
    if ($doReuse)
    {
        foreach my $i (0 .. $#instructs)
        {
            if ($i & 3)
            {
                if (exists $instructs[$i]{reuse})
                {
                    my $reuse = 0;
                    $reuse |= $reuseSlots{$_} foreach keys %{$instructs[$i]{reuse}};

                    # Overwrite parsed value
                    $ctrl->{reuse}[($i & 3) - 1] = $reuse;
                }
            }
            else
                { $ctrl = $instructs[$i]; }
        }
    }

    # final pass to piece together control codes
    my @codes;
    foreach my $i (0 .. $#instructs)
    {
        if ($i & 3)
        {
            push @codes, $instructs[$i]{code};
        }
        else
        {
            my ($ctrl, $ruse) = @{$instructs[$i]}{qw(ctrl reuse)};
            push @codes,
                ($ctrl->[0] <<  0) | ($ctrl->[1] << 21) | ($ctrl->[2] << 42) | # ctrl codes
                ($ruse->[0] << 17) | ($ruse->[1] << 38) | ($ruse->[2] << 59);  # reuse codes
        }
    }

    # return the kernel data
    return {
        RegCnt => $regCnt,
        BarCnt => $barCnt,
        InsCnt => $insCnt,
        KernelData => \@codes,
    };
}

# Useful for testing op code coverage of existing code, extracting new codes and flags
sub Test
{
    my ($fh, $all) = @_;

    my @instructs;
    my ($pass, $fail) = (0,0);

    while (my $line = <$fh>)
    {
        my (@ctrl, @reuse);

        next unless processSassCtrlLine($line, \@ctrl, \@reuse);

        foreach my $fileReuse (@reuse)
        {
            $line = <$fh>;

            my $inst = processSassLine($line) or next;

            $inst->{reuse} = $fileReuse;
            my $fileCode = $inst->{code};

            if (exists $relOffset{$inst->{op}})
            {
                # these ops need to be converted from absolute addresses to relative in the sass output by cuobjdump
                $inst->{inst} =~ s/(0x[0-9a-f]+)/sprintf '0x%06x', ((hex($1) - $inst->{num} - 8) & 0xffffff)/e;
            }

            my $match = 0;
            foreach my $gram (@{$grammar{$inst->{op}}})
            {
                if ($inst->{inst} =~ $gram->{rule})
                {
                    my @caps;
                    my ($code, $reuse) = genCode($inst->{op}, $gram, \@caps);

                    $inst->{caps}      = join ', ', sort @caps;
                    $inst->{codeDiff}  = $fileCode  ^ $code;
                    $inst->{reuseDiff} = $fileReuse ^ $reuse;

                    # compare calculated and file values
                    if ($code == $fileCode && $reuse == $fileReuse)
                    {
                        $inst->{grade} = 'PASS';
                        push @instructs, $inst if $all;
                        $pass++;
                    }
                    else
                    {
                        $inst->{grade} = 'FAIL';
                        push @instructs, $inst;
                        $fail++;
                    }


                    $match = 1;
                    last;
                }
            }
            unless ($match)
            {
                $inst->{grade}     = 'FAIL';
                $inst->{codeDiff}  = $fileCode;
                $inst->{reuseDiff} = $fileReuse;
                $fail++;
            }
        }
    }
    my %maxLen;
    foreach (@instructs)
    {
        $maxLen{$_->{op}} = length($_->{ins}) if length($_->{ins}) > $maxLen{$_->{op}};
    }
    my ($lastOp, $template);
    foreach my $inst (sort {
        $a->{op}        cmp $b->{op}        ||
        $a->{codeDiff}  <=> $b->{codeDiff}  ||
        $a->{reuseDiff} <=> $b->{reuseDiff} ||
        $a->{ins}       cmp $b->{ins}
        } @instructs)
    {
        if ($lastOp ne $inst->{op})
        {
            $lastOp   = $inst->{op};
            $template = "%s 0x%016x %x 0x%016x %x %5s%-$maxLen{$lastOp}s   %s\n";
            printf "\n%s %-18s %s %-18s %s %-5s%-$maxLen{$lastOp}s   %s\n", qw(Grad OpCode R opCodeDiff r Pred Instruction Captures);
        }
        printf $template, @{$inst}{qw(grade code reuse codeDiff reuseDiff pred ins caps)};
    }
    print "\nTotals: Pass: $pass Fail: $fail\n";
    return $fail;
}

# Convert cuobjdump sass to the working format
sub Extract
{
    my ($in, $out) = @_;

    my %labels;
    my $labelnum = 1;

    my @data;
    FILE: while (my $line = <$in>)
    {
        my (@ctrl, @ruse);
        next unless processSassCtrlLine($line, \@ctrl, \@ruse);

        CTRL: foreach my $ctrl (@ctrl)
        {
            $line = <$in>;

            my $inst = processSassLine($line) or next CTRL;

            # Convert branch/jump/call addresses to labels
            if ((exists($relOffset{$inst->{op}}) || exists($absOffset{$inst->{op}})) && $inst->{inst} =~ m'(0x[0-9a-f]+)')
            {
                my $target = hex($1);

                # skip the final BRA and stop processing the file
                last FILE if $inst->{op} eq 'BRA' && $target == $inst->{num};

                # check to see if we've already generated a label for this target address
                my $label = $labels{$target};
                unless ($label)
                {
                    # generate a label name and cache it
                    $label = $labels{$target} = "TARGET$labelnum";
                    $labelnum++;
                }
                # replace address with name
                $inst->{inst} =~ s/(0x[0-9a-f]+)/$label/;
            }
            $inst->{ctrl} = printCtrl($ctrl);

            push @data, $inst;
        }
    }
    # make a second pass now that we have the complete instruction address to label mapping
    foreach my $inst (@data)
    {
        print $out "$labels{$inst->{num}}:\n" if exists $labels{$inst->{num}};
        printf $out "%s %5s%s\n", @{$inst}{qw(ctrl pred inst)};
    }
}

sub Preprocess
{
    my ($file, $doReg) = @_;

    # Strip out comments
    $file =~ s|^<COMMENT>.*?^</COMMENT>\n?||gms;

    # Pull in the reg map first as the Scheduler will need it to handle vector instructions
    $file =~ m'^<REGISTER_MAPPING>(.*?)^</REGISTER_MAPPING>'ms;

    my $regMap = getRegisterMap($file, $1);

    # Execute the CODE sections
    $file =~ s/^<CODE>(.*?)^<\/CODE>\n?/my $r = eval "package MaxAs::CODE; $1"; $r || die "CODE:\n$1\n\nError: $@\n"/egms;

    # XMAD macros:
    $file = replaceXMADs($file);

    # Pick out the SCHEDULE_BLOCK sections
    my @schedBlocks = $file =~ m'^<SCHEDULE_BLOCK>(.*?)^</SCHEDULE_BLOCK>'gms;

    # Schedule them
    foreach my $i (0 .. $#schedBlocks)
    {
        $schedBlocks[$i] = Scheduler($schedBlocks[$i], $i+1, $regMap);
    }

    # Replace the results
    $file =~ s|^<SCHEDULE_BLOCK>(.*?)^</SCHEDULE_BLOCK>\n?| shift @schedBlocks |egms;

    # Do the regmapping stage if requested
    if ($doReg && $file =~ s|^<REGISTER_MAPPING>(.*?)^</REGISTER_MAPPING>\n?||ms)
    {
        my $out;
        foreach my $line (split "\n", $file)
        {
            # skip comment lines
            if ($line !~ m'^\s*(?:#|//)' && $line !~ m'^\s*$')
            {
                # Avoid replacing the c in a constant load..
                # TODO: there are probably other annoying exceptions that needed to be added here.
                $line =~ s/(\w+)(?!\s*\[)/exists $regMap->{$1} ? $regMap->{$1} : $1/eg;
            }
            $out .= "$line\n";
        }
        $file = $out;
    }

    return $file;
}

# break the registers down into source and destination categories for the scheduler
my %srcReg   = map { $_ => 1 } qw(r8 r20 r39 p12 p29 p39);
my %destReg  = map { $_ => 1 } qw(r0 p0 p3 p45 p48);
my %regops   = (%srcReg, %destReg);
my @itypes   = qw(class lat rlat tput dual);

sub Scheduler
{
    my ($block, $blockNum, $regMap) = @_;

    # Build a reverse lookup of reg numbers to names.
    # Note that this may not be unique, so include multiples in a list
    my %mapReg;
    foreach my $regName (keys %$regMap)
    {
        push @{$mapReg{$regMap->{$regName}}}, $regName;
    }

    my $lineNum = 0;

    my (@instructs, @comments);
    foreach my $line (split "\n", $block)
    {
        # keep track of line nums in the physical file
        $lineNum++;

        unless (preProcessLine($line))
        {
            push @comments, $line if $line =~ m'\S';
            next;
        }

        # match an instruction
        if (my $inst = processAsmLine($line, $lineNum))
        {
            $inst->{exeTime} = 0;
            push @instructs, $inst;
        }
        # match a label
        elsif ($line =~ m'^([a-zA-Z]\w*):')
        {
            die "SCHEDULE_BLOCK's cannot contain labels. block: $blockNum line: $lineNum\n";
        }
        else
        {
            die "badly formed line at block: $blockNum line: $lineNum: $line\n";
        }
    }

    my (%writes, %reads, @ready, @schedule);
    # assemble the instructions to op codes
    foreach my $instruct (@instructs)
    {
        my $match = 0;
        foreach my $gram (@{$grammar{$instruct->{op}}})
        {
            if ($instruct->{inst} =~ $gram->{rule})
            {
                my (@dest, @src);

                # copy over instruction types for easier access
                @{$instruct}{@itypes} = @{$gram->{type}}{@itypes};

                # A predicate prefix is treated as a source reg
                push @src, $instruct->{predReg} if $instruct->{pred};

                # Populate our register source and destination lists, skipping any zero or true values
                foreach my $operand (grep { exists $regops{$_} } keys %+)
                {
                    # figure out which list to populate
                    my $list = exists($destReg{$operand}) && !exists($noDest{$instruct->{op}}) ? \@dest : \@src;

                    # Filter out RZ and PT
                    my $badVal = substr($operand,0,1) eq 'r' ? 'RZ' : 'PT';

                    if ($+{$operand} ne $badVal)
                    {
                        # add the value to list with the correct prefix
                        push @$list, $+{$operand};

                        if ($operand eq 'r0')
                        {
                            # map the reg name to it's number and get the corresponding vector registers if appropriate
                            my @r = getVecRegisters($regMap->{$+{r0}}, $+{type});

                            # We need to add the linked registers to the list for vector instructions
                            if (@r > 1)
                            {
                                # first one already added
                                shift @r;

                                # Now do an inverse lookup of the number to name mapping and add each to the list.
                                push @$list, @{$mapReg{$_}} foreach @r;
                            }
                        }
                    }
                }

                # Find Read-After-Write dependencies
                foreach my $src (grep { exists $writes{$_} } @src)
                {
                    # Memory operations get delayed access to registers but not to the predicate
                    my $regLatency = $src eq $instruct->{predReg} ? 0 : $instruct->{rlat};

                    # the parent should be the most recently added dest op to the stack
                    foreach my $parent (@{$writes{$src}})
                    {
                        # add this instruction as a child of the parent
                        # set the edge to the total latency of reg source availability
                        push @{$parent->{children}}, [$instruct, $parent->{lat} - $regLatency];
                        $instruct->{parents}++;

                        # if the destination was conditionally executed, we also need to keep going back till it wasn't
                        last unless $parent->{pred};
                    }
                }

                # Find Write-After-Read dependencies
                foreach my $dest (grep { exists $reads{$_} } @dest)
                {
                    # Flag this instruction as dependent to any previous read
                    foreach my $reader (@{$reads{$dest}})
                    {
                        # no need to stall for these types of dependencies
                        push @{$reader->{children}}, [$instruct, 0];
                        $instruct->{parents}++;
                    }
                    # Once dependence is marked we can clear out the read list (unless this write was conditional).
                    # The assumption here is that you would never want to write out a register without
                    # subsequently reading it in some way prior to writing it again.
                    delete $reads{$dest} unless $instruct->{pred};
                }

                # For a dest reg, push it onto the write stack
                unshift @{$writes{$_}}, $instruct foreach @dest;

                # For a src reg, push it into the read list
                push @{$reads{$_}}, $instruct foreach @src;

                # if this instruction has no dependencies it's ready to go
                push @ready, $instruct if !exists $instruct->{parents};

                $match = 1;
                last;
            }
        }
        die "Unable to recognize instruction at block: $blockNum line: $lineNum: $instruct->{inst}\n" unless $match;
    }
    %writes = ();
    %reads  = ();

    if (@ready)
    {
        # update dependent counts for sorting hueristic
        my $readyParent = { children => [ map { [ $_, 1 ] } @ready ] };
        countUniqueDescendants($readyParent);
        updateDepCounts($readyParent);

        # sort the initial ready list
        @ready = sort {
            $b->{deps}    <=> $a->{deps}   ||
            $a->{lineNum} <=> $b->{lineNum}
            } @ready;
    }

    # Process the ready list, adding new instructions to the list as we go.
    my $clock = 0;
    while (my $instruct = shift @ready)
    {
        my $stall = $instruct->{stall};

        # apply the stall to the previous instruction
        if (@schedule && $stall < 16)
        {
            my $prev = $schedule[$#schedule];

            # if stall is greater than 4 then also yield
            $prev->{ctrl} &= $stall > 4 ? 0x1ffe0 : 0x1fff0;
            $prev->{ctrl} |= $stall;
            $clock += $stall;
        }
        # For stalls bigger than 15 we assume the user is managing it with a barrier
        else
        {
            $instruct->{ctrl} &= 0x1fff0;
            $instruct->{ctrl} |= 1;
            $clock += 1;
        }
        print "$clock: $instruct->{inst}\n" if $DEBUG;

        # add a new instruction to the schedule
        push @schedule, $instruct;

        # update each child with a new earliest execution time
        if (my $children = $instruct->{children})
        {
            foreach (@$children)
            {
                my ($child, $latency) = @$_;

                # update the earliest clock value this child can safely execute
                my $earliest = $clock + $latency;
                $child->{exeTime} = $earliest if $child->{exeTime} < $earliest;

                print "\t\t$child->{exeTime},$child->{parents} $child->{inst}\n" if $DEBUG;

                # decrement parent count and add to ready queue if none remaining.
                push @ready, $child if --$child->{parents} < 1;
            }
            delete $instruct->{children};
        }

        # update stall and mix values in the ready queue on each iteration
        foreach my $ready (@ready)
        {
            # calculate how many instructions this would cause the just added instruction to stall.
            $stall = $ready->{exeTime} - $clock;
            $stall = 1 if $stall < 1;

            # if using the same compute resource as the prior instruction then limit the throughput
            if ($ready->{class} eq $instruct->{class})
            {
                $stall = $ready->{tput} if $stall < $ready->{tput};
            }
            # dual issue with a simple instruction (tput == 1)
            elsif ($ready->{dual} && !$instruct->{dual} && $instruct->{tput} == 1 && $stall == 1)
            {
                $stall = 0;
            }
            $ready->{stall} = $stall;

            # add an instruction class mixing huristic that catches anything not handled by the stall
            $ready->{mix} = $ready->{class} ne $instruct->{class} || 0;
        }

        # sort the ready list by stall time, mixing huristic, dependencies and line number
        @ready = sort {
            $a->{stall}   <=> $b->{stall}  ||
            $b->{mix}     <=> $a->{mix}    ||
            $b->{deps}    <=> $a->{deps}   ||
            $a->{lineNum} <=> $b->{lineNum}
            } @ready;

        if ($DEBUG)
        {
            print  "\text,stl,mix,dep,lin, inst\n";
            printf "\t%3s,%3s,%3s,%3s,%3s, %s\n", @{$_}{qw(exeTime stall mix deps lineNum inst)} foreach @ready;
        }
    }

    my $out;
    $out .= "$_\n" foreach @comments;
    $out .= join('', printCtrl($_->{ctrl}), @{$_}{qw(space inst comment)}, "\n") foreach @schedule;
    return $out;
}

sub getRegisterMap
{
    my ($file, $regmapText) = @_;

    my %regMap;
    foreach my $line (split "\n", $regmapText)
    {
        # strip leading space
        $line =~ s|^\s+||;
        # strip comments
        $line =~ s{(?:#|//).*}{};
        # strip trailing space
        $line =~ s|\s+$||;
        # skip blank lines
        next unless $line =~ m'\S';

        my ($regNums, $regNames) = split '\s*:\s*', $line;

        my (@numList, @nameList);
        foreach (split '\s*,\s*', $regNums)
        {
            my ($start, $stop) = split '\s*\-\s*';
            die "Bad register number: $_ at: $line\n" if grep m'\D', $start, $stop;
            push @numList, defined($stop) ? ($start .. $stop) : ($start);
        }
        foreach (split '\s*,\s*', $regNames)
        {
            if (m'^(\w+)<((?:\d+\s*\-\s*\d+\s*\|?\s*)+)>(\w*)$')
            {
                my ($name1, $name2) = ($1, $3);
                foreach (split '\s*\|\s*', $2)
                {
                    my ($start, $stop) = split '\s*\-\s*';
                    push @nameList, "$name1$_$name2" foreach ($start .. $stop);
                }
            }
            elsif (m'^\w+$')
            {
                push @nameList, $_;
            }
            else
            {
                die "Bad register name: '$_' at: $line\n";
            }
        }
        die "Missmatched register mapping at: $line\n" if @numList < @nameList;

        $regMap{$nameList[$_]} = sprintf('R%d',$numList[$_]) foreach (0..$#nameList);
    }
    return \%regMap;
}


my $CtrlRe = qr'(?<ctrl>[0-9a-fA-F\-]{2}:[1-6\-]:[1-6\-]:[\-yY]:[0-9a-fA-F])';
my $PredRe = qr'(?<pred>@!?(?<predReg>P\d)\s+)';
my $InstRe = qr"$PredRe?(?<op>\w+)(?<rest>[^;]*;)"o;
my $CommRe = qr'(?<comment>.*)';

sub processAsmLine
{
    my ($line, $lineNum) = @_;

    if ($line =~ m"^$CtrlRe(?<space>\s+)$InstRe$CommRe"o)
    {
        return {
            lineNum => $lineNum,
            pred    => $+{pred},
            predReg => $+{predReg},
            space   => $+{space},
            op      => $+{op},
            comment => $+{comment},
            inst    => normalizeSpacing($+{pred} . $+{op} . $+{rest}),
            ctrl    => readCtrl($+{ctrl}, $line),
        };
    }
    return undef;
}

sub processSassLine
{
    my $line = shift;

    if ($line =~ m"^\s+/\*(?<num>[0-9a-f]+)\*/\s+$InstRe\s+/\* (?<code>0x[0-9a-f]+)"o)
    {
        return {
            num     => hex($+{num}),
            pred    => $+{pred},
            op      => $+{op},
            ins     => normalizeSpacing($+{op} . $+{rest}),
            inst    => normalizeSpacing($+{pred} . $+{op} . $+{rest}),
            code    => hex($+{code}),
        };
    }
    return undef;
}

sub processSassCtrlLine
{
    my ($line, $ctrl, $ruse) = @_;

    return 0 unless $line =~ m'^\s+\/\* (0x[0-9a-f]+)';

    my $code = hex($1);
    if (ref $ctrl)
    {
        push @$ctrl, ($code & 0x000000000001ffff) >> 0;
        push @$ctrl, ($code & 0x0000003fffe00000) >> 21;
        push @$ctrl, ($code & 0x07fffc0000000000) >> 42;
    }
    if (ref $ruse)
    {
        push @$ruse, ($code & 0x00000000001e0000) >> 17;
        push @$ruse, ($code & 0x000003c000000000) >> 38;
        push @$ruse, ($code & 0x7800000000000000) >> 59;
    }
    return 1;
}

sub replaceXMADs
{
    my $file = shift;

    # XMAD.LO d, a, b, c, x;
    # ----------------------
    # XMAD.MRG x, a, b.H1, RZ;
    # XMAD d, a, b, c;
    # XMAD.PSL.CBCC d, a.H1, x.H1, d;
    $file =~ s/\n$CtrlRe(?<space>\s+)($PredRe)?XMAD\.LO\s+(?<d>\w+)\s*,\s*(?<a>\w+)\s*,\s*(?<b>\w+)\s*,\s*(?<c>\w+)\s*,\s*(?<x>\w+)\s*;$CommRe/

        sprintf '
%1$s%2$s%3$sXMAD.MRG %8$s, %5$s, %6$s.H1, RZ;%9$s
%1$s%2$s%3$sXMAD %4$s, %5$s, %6$s, %7$s;
%1$s%2$s%3$sXMAD.PSL.CBCC %4$s, %5$s.H1, %8$s.H1, %4$s;',
        @+{qw(ctrl space pred d a b c x comment)}
    /egmos;

    #TODO: add more XMAD macros
    return $file;
}

# map binary control notation on to easier to work with format.
sub printCtrl
{
    my $code = shift;

    my $stall = ($code & 0x0000f) >> 0;
    my $yield = ($code & 0x00010) >> 4;
    my $wrtdb = ($code & 0x000e0) >> 5;  # write dependency barier
    my $readb = ($code & 0x00700) >> 8;  # read  dependency barier
    my $watdb = ($code & 0x1f800) >> 11; # wait on dependency barier

    $yield = $yield ? '-' : 'Y';
    $wrtdb = $wrtdb == 7 ? '-' : $wrtdb + 1;
    $readb = $readb == 7 ? '-' : $readb + 1;
    $watdb = $watdb ? sprintf('%02x', $watdb) : '--';

    return sprintf '%s:%s:%s:%s:%x', $watdb, $readb, $wrtdb, $yield, $stall;
}
sub readCtrl
{
    my ($ctrl, $context) = @_;
    my ($watdb, $readb, $wrtdb, $yield, $stall) = split ':', $ctrl;

    $watdb = $watdb eq '--' ? 0 : hex $watdb;
    $readb = $readb eq '-'  ? 7 : $readb - 1;
    $wrtdb = $wrtdb eq '-'  ? 7 : $wrtdb - 1;
    $yield = $yield eq 'y' || $yield eq 'Y'  ? 0 : 1;
    $stall = hex $stall;

    die sprintf('wait dep out of range(0x00-0x3f): %x at %s',   $watdb, $context) if $watdb != ($watdb & 0x3f);

    return
        $watdb << 11 |
        $readb << 8  |
        $wrtdb << 5  |
        $yield << 4  |
        $stall << 0;
}

sub getVecRegisters
{
    my ($reg, $type) = @_;
    my $base = substr $reg, 1;
    my $cnt = $type eq '.64' ? 1 : $type eq '.128' ? 3 : 0;
    return map "R$_", $base .. $base+$cnt;
}

# convert extra spaces to single spacing to make our re's simplier
sub normalizeSpacing
{
    my $inst = shift;
    $inst =~ s/\t/ /g;
    $inst =~ s/\s{2,}/ /g;
    return $inst;
}

sub preProcessLine
{
    # strip leading space
    $_[0] =~ s|^\s+||;

    # preserve comment but check for emptiness
    my $val = shift;

    # strip comments
    $val =~ s{(?:#|//).*}{};

    # skip blank lines
    return $val =~ m'\S';
}

# traverse the graph and count total descendants per node.
# only count unique nodes (by lineNum)
sub countUniqueDescendants
{
    my $node = shift;
    if (my $children = $node->{children})
    {
        foreach my $child (grep $_->[1], @$children) # skip WaR deps
        {
            $node->{deps}{$_}++ foreach countUniqueDescendants($child->[0]);
        }
    }
    else
    {
        return $node->{lineNum};
    }
    return ($node->{lineNum}, keys %{$node->{deps}});
}
# convert hash to count for easier sorting.
sub updateDepCounts
{
    my $node = shift;
    if (my $children = $node->{children})
    {
        updateDepCounts($_->[0]) foreach @$children;
    }
    $node->{deps} = ref $node->{deps} ? keys %{$node->{deps}} : $node->{deps}+0;
}

__END__



