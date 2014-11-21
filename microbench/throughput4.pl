#!/usr/bin/perl
use strict;

my $loopSize  = 512;
my $blocks    = 64;
my $loops     = 102400;
my $fileName  = 'throughput2.sass';

writeSassFile($fileName, $loops);

#print `maxas.pl -p $fileName`;
#exit;

print `maxas.pl -i $fileName microbench.cubin`;
exit if $?;

foreach my $thread128 (4)
{
    my $threads   = $thread128 * 128;
    my $fops      = 2 * $loops * $loopSize * $blocks * $threads;

    print "./microbench e $blocks $threads $fops\n\n";
    my $data = `./microbench e $blocks $threads $fops`;
    exit($?) if $?;

    my ($gflops) = $data =~ /GFLOPS: ([0-9]+)/ms;

    printf "%d %d %d %.2f\n", $thread128, $threads, $gflops, 100 * $gflops / 3050.0;
}

exit;

sub writeSassFile
{
    my ($filename, $loops) = @_;

    open my $fh, ">$filename" or die "$filename: $!";

    printf $fh <<'EOF', $loops;
# Kernel: microbench

<REGISTER_MAPPING>

    0-10 : result, r1, r2, r3
    20-27 ~ count, stop

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

    foreach my $i (0 .. 511)
    {
        my $yield = ($i + 32) & 63 ? '-' : 'Y';

        my $stall = $i == 511 ? 0 : 1;

        #$out .= "--:-:-:$yield:1      FFMA r3, r1, r2, r3;\n";
        #$out .= "--:-:-:-:1      FFMA r3, r1, r2, r3;\n";
        #$out .= "--:-:-:-:1      FFMA r3, r1, r2, r3;\n";
        #$out .= "--:-:-:-:0      FFMA r3, r1, r2, r3;\n";
        #$out .= "--:-:-:-:1      I2F.F32.S16 result, r1;\n";

        #$out .= "--:-:-:$yield:$stall      VADD.S16.S16.SAT.MRG_16L result, r1, r2, RZ;\n";
        #$out .= "--:-:-:-:1      MOV result, RZ;\n";

        $out .= "--:-:-:$yield:$stall      IADD.SAT result, r1, r2;\n";
        #$out .= "--:-:-:$yield:$stall      VMAD.S8.S8.SAT result, r1, r2, r3;\n";
        #$out .= "--:-:-:$yield:$stall      XMAD result, r1, r2, r3;\n";
    }
    return $out;
</CODE>

--:-:-:Y:5  @P0 BRA LOOP;
--:-:-:-:5      EXIT;
EOF

    close $fh;
}

__END__

VMAD.U8.U8

dddd 2655 / 4968 = 53.4%
1d1d 4594 / 4968 = 92.4%
11d  4746 / 4968 = 95.5%
111d 4841 / 4968 = 97.4%

block context switches are a little more expensive than thread context switches

stall codes:

f : 13 clocks
e :  8 clocks
d :  6 clocks
c :  8 clocks, no yield
b : 11 clocks
a : 10 clocks
9 :  9 clocks
8 :  8 clocks
7 :  7 clocks
6 :  6 clocks
5 :  5 clocks
4 :  4 clocks
3 :  3 clocks
2 :  2 clocks
1 :  1 clocks,  no yield
0 :  0 clocks,  no yield, dual issue