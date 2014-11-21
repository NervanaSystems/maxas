#!/usr/bin/perl
use strict;
my %p;

$p{N}         = 8192;
$p{blocking}  = 8;
$p{unroll}    = 8;
$p{threads}   = 64;   #256

$p{csize}     = $p{blocking} * $p{blocking};
$p{loopSize}  = $p{unroll} * $p{csize};
$p{width}     = sqrt($p{csize} * $p{threads});
$p{blocks}    = ($p{N} / $p{width}) * ($p{N} / $p{width});
$p{loops}     = $p{N} / $p{unroll};
$p{fops}      = 2 * $p{loops} * $p{loopSize} * $p{blocks} * $p{threads};

my $fileName  = 'throughput2.sass';

my @params = qw(N blocking unroll threads csize loopSize loops width blocks fops);

#print join("\t", @params), "\n";
#print join("\t", @p{@params}), "\n";

print map sprintf("%-9s: %d\n", $_, $p{$_}), @params;

writeSassFile($fileName, $p{loopSize}, $p{loops});

#print `maxas.pl -p $fileName`;
#exit;

print `maxas.pl -i $fileName microbench.cubin`;

exit if $?;

my $data = `Release\\microbench.exe e $p{blocks} $p{threads} $p{fops} 50`;

my ($gflops) = $data =~ /GFLOPS: ([0-9]+)/ms;

print $data;

#printf "%d %4d %4d %d\n", $thread128, $loopSize, $loops, $gflops;




sub writeSassFile
{
    my ($filename, $loopSize, $loops) = @_;

    open my $fh, ">$filename" or die "$filename: $!";

    printf $fh <<'END_SASS', $loops;
# Kernel: microbench

<REGISTER_MAPPING>

     1, 9, 2,10,17,25,18,26 : cy0x<0-7>
     5,13, 6,14,21,29,22,30 : cy1x<0-7>
     3,11, 0, 8,19,27,16,24 : cy2x<0-7>
     7,15, 4,12,23,31,20,28 : cy3x<0-7>
    35,43,32,40,51,59,48,56 : cy4x<0-7>
    39,47,36,44,55,63,52,60 : cy5x<0-7>
    33,41,34,42,49,57,50,58 : cy6x<0-7>
    37,45,38,46,53,61,54,62 : cy7x<0-7>

    64-71   : j0Ax<0-3>, j0By<0-3>
    72-79   : j1Ax<0-3>, j1By<0-3>

    0-79 : r<0-79>

    100-101 : count, stop

    //102-112 ~ readAs, readBs, writeS

</REGISTER_MAPPING>

--:-:-:-:1      MOV count, RZ;
--:-:-:-:1      MOV32I stop, %d;
//--:-:-:-:1      MOV writeS, RZ;
//--:-:-:-:1      MOV readAs, RZ;
//--:-:-:-:1      MOV readBs, RZ;

<CODE>
    return join '', map "--:-:-:-:1      MOV r$_, RZ;\n", 0..63;
</CODE>

<CODE>
    return join '', map "--:-:-:-:1      MOV32I r$_, 0x00010001;\n", 64..79;
</CODE>

LOOP:

--:-:-:-:1      ISETP.LE.AND P0, PT, count, stop, PT;
--:-:-:-:1      IADD count, count, 1;

<CODE>
    my $out;

    my @swirl1 = ([0,0],[0,4],[4,4],[4,0]);
    my @swirl2 = ([0,0],[1,0],[1,1],[0,1]);
    my @swirl3 = ([0,2],[2,2],[2,0],[0,0]);

    my @cOrder;
    foreach my $s1 (@swirl1)
    {
        foreach my $s2 (@swirl2)
        {
            foreach my $s3 (@swirl3)
            {
                push @cOrder, [$s1->[0] + $s2->[0] + $s3->[0], $s1->[1] + $s2->[1] + $s3->[1]];
            }
        }
    }

    foreach my $j (0..7)
    {
        my $odd  = $j & 1;
        my $nOdd = !$odd + 0;

        my %%insert;

        #$insert{c62} = "01:-:-:-:5      BAR.SYNC 0;\n" if $j == 6;

        $insert{c62} =
                "--:-:-:-:1      LOP.XOR readAs, readAs, 0;\n" .
                "--:-:-:-:1      LOP.XOR readBs, readBs, 0;\n" .
                "--:-:-:-:1      LOP.XOR readAs, readAs, 0;\n" .
                "--:-:-:-:1      LOP.XOR readBs, readBs, 0;\n" .
                "--:-:-:-:1      LOP.XOR writeS, writeS, 0;\n" if $j == 8;

        foreach my $c (0 .. 63)
        {
            my ($x,$y) = @{$cOrder[$c]};
            my $ins    = $insert{"c$c"} || '';
            my $stall  = ($c == 63 && $j == 7) ? 0 : 1; #1; #$ins ||
            my $yield  = $c == 32 ? 'Y' : '-';
            my $wait   = '--'; #$c ? '--' : '01';

            my $xReg  = $x >> 1;
            my $yReg  = $y >> 1;
            my $xPart = $x & 1 ? '.H1' : '';
            my $yPart = $y & 1 ? '.H1' : '';

            $out .= sprintf "$wait:-:-:$yield:$stall      XMAD cy%%dx%%d, j%%dAx%%d%%s, j%%dBy%%d%%s, cy%%dx%%d;\n%%s", $y,$x,  $odd,$xReg,$xPart,  $odd,$yReg,$yPart,  $y,$x,  $ins;
        }
    }
    return $out;
</CODE>

--:-:-:Y:5  @P0 BRA LOOP;
--:-:-:-:5      EXIT;
END_SASS

    close $fh;
}

__END__

        my %%insert = (
            c0 => "--:-:-:-:1      LDS.U.128 j${nOdd}Ax00, [readAs+0x10];\n",
            c2 => "--:-:-:-:1      LDS.U.128 j${nOdd}By00, [readBs+0x10];\n",
            c4 => "--:-:-:-:1      LDS.U.128 j${nOdd}Ax64, [readAs+0x10];\n",
            c6 => "--:-:1:-:1      LDS.U.128 j${nOdd}By64, [readBs+0x10];\n",
        );