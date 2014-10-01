#!/usr/bin/perl
use strict;

my $loopSize  = 512;
my $blocks    = 32;
my $loops     = 10240000;
my $fileName  = 'throughput2.sass';

writeSassFile($fileName, $loops);

#print `maxas.pl -p $fileName`;
#exit;

print `maxas.pl -i $fileName microbench.cubin`;
exit if $?;

foreach my $thread128 (2)
{
    my $threads   = $thread128 * 128;
    my $fops      = 2 * $loops * $loopSize * $blocks * $threads;

    my $data = `Release\\microbench.exe e $blocks $threads $fops`;

    my ($gflops) = $data =~ /GFLOPS: ([0-9]+)/ms;

    printf "%d %d %d\n", $thread128, $threads, $gflops;
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

        $out .= "--:-:-:$yield:$stall      FFMA result, r1, r2, r3;\n";
    }
    return $out;
</CODE>

--:-:-:Y:5  @P0 BRA LOOP;
--:-:-:-:5      EXIT;
EOF

    close $fh;
}

__END__

