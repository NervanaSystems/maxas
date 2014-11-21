#!/usr/bin/perl
use strict;

my %data;

foreach my $thread128 (1 .. 8)
{
    foreach my $size64 (8 .. 16)
    {
        my $loopSize  = $size64 * 64;
        my $loops     = int(2 * 1638400 / ($size64 * $thread128));

        my $blocks    = 16;
        my $threads   = $thread128 * 128;
        my $fops      = 2 * $loops * $loopSize * $blocks * $threads;
        my $fileName  = 'throughput2.sass';

        #printf "%d %4d %4d %d\n", $thread128, $loopSize, $loops, $fops;
        #next;

        writeSassFile($fileName, $loopSize, $loops);

        `maxas.pl -i $fileName microbench.cubin`;

        exit if $?;

        my $data = `Release\\microbench.exe e $blocks $threads $fops`;

        my ($gflops) = $data =~ /GFLOPS: ([0-9]+)/ms;

        printf "%d %4d %4d %d\n", $thread128, $loopSize, $loops, $gflops;

        push @{$data{$loopSize}}, $gflops;
    }
}
print join("\t", 'size', 1 .. 8), "\n";
foreach my $loopSize (sort {$a <=> $b} keys %data)
{
    print join("\t", $loopSize, @{$data{$loopSize}}), "\n";
}

exit;

sub writeSassFile
{
    my ($filename, $loopSize, $loops) = @_;

    open my $fh, ">$filename" or die "$filename: $!";

    printf $fh <<'EOF', $loops, $loopSize, $loopSize;
# Kernel: microbench

<REGISTER_MAPPING>

    0-10 : result, r1, r2, r3, count, stop

</REGISTER_MAPPING>

--:-:-:-:1      MOV count, RZ;
--:-:-:-:1      MOV32I stop, %d;
--:-:-:-:1      MOV32I r1, 1.0;
--:-:-:-:1      MOV32I r2, 1.0;
--:-:-:-:4      MOV32I r3, 1.0;

LOOP:

--:-:-:-:1      ISETP.LE.AND P0, PT, count, stop, PT;
--:-:-:-:1      IADD count, count, 1;

<CODE>
    my $out;

    foreach my $i (0 .. %d)
    {
        my $y = %d > 64 && (($i + 32) & 63) ? '-' : 'Y';

        $out .= "--:-:-:$y:1      FFMA result, r1, r2, r3;\n";
    }
    return $out;
</CODE>

--:-:-:Y:5  @P0 BRA LOOP;
--:-:-:-:5      EXIT;
EOF

    close $fh;
}

__END__

