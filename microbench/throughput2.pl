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

     3, 2,11,10,19,18,27,26 : cx00y<00-03|64-67>
     7, 6,15,14,23,22,31,30 : cx01y<00-03|64-67>
     1, 0, 9, 8,17,16,25,24 : cx02y<00-03|64-67>
     5, 4,13,12,21,20,29,28 : cx03y<00-03|64-67>
    35,34,43,42,51,50,59,58 : cx64y<00-03|64-67>
    39,38,47,46,55,54,63,62 : cx65y<00-03|64-67>
    33,32,41,40,49,48,57,56 : cx66y<00-03|64-67>
    37,36,45,44,53,52,61,60 : cx67y<00-03|64-67>

    64-79 : j0Ax<00-03|64-67>, j0By<00-03|64-67>
    80-95 : j1Ax<00-03|64-67>, j1By<00-03|64-67>

    0-127 : r<0-127>

    100-101 : count, stop

    //102-112 ~ readAs, readBs, writeS

</REGISTER_MAPPING>

--:-:-:-:1      MOV count, RZ;
--:-:-:-:1      MOV32I stop, %d;
//--:-:-:-:1      MOV writeS, RZ;
//--:-:-:-:1      MOV readAs, RZ;
//--:-:-:-:1      MOV readBs, RZ;

<CODE>
    return join '', map "--:-:-:-:1      MOV32I r$_, 1.0;\n", 0..95;
</CODE>

LOOP:

--:-:-:-:1      ISETP.LE.AND P0, PT, count, stop, PT;
--:-:-:-:1      IADD count, count, 1;

<CODE>
    my $out;


    my @cOrder;
    #my @swirl = ([0,1],[0,0],[2,0],[2,1]);
    my @swirl = ([2,0],[2,1],[0,1],[0,0]);
    #my @swirl = ([0,1],[0,0],[1,0],[1,1]);
    my @xVals = (0,1,64,65);
    #my @xVals = (0,2,64,66);

    my @yVals = (0,2,64,66);

    foreach my $y (@yVals)
    {
        foreach my $x (@xVals)
        {
            push @cOrder, sprintf('x%%02dy%%02d', $x + $_->[0], $y + $_->[1]) foreach @swirl;
        }
        @xVals = reverse @xVals;
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
            my ($x,$y) = $cOrder[$c] =~ /^(x\d+)(y\d+)/;
            my $ins    = $insert{"c$c"} || '';
            my $stall  = ($c == 63 && $j == 7) ? 0 : 1; #1; #$ins ||
            my $yield  = $c == 32 ? 'Y' : '-';
            my $wait   = '--'; #$c ? '--' : '01';

            $out .= "$wait:-:-:$yield:$stall      FFMA c$cOrder[$c], j${odd}A$x, j${odd}B$y, c$cOrder[$c];\n$ins";
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