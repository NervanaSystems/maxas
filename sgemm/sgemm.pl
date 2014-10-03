#!/usr/bin/perl
use strict;

my $CU_AD_FORMAT_UNSIGNED_INT8  = 0x01;
my $CU_AD_FORMAT_UNSIGNED_INT16 = 0x02;
my $CU_AD_FORMAT_FLOAT          = 0x20;

if (!-f 'sgemm_pre_128.sass' || (stat 'sgemm128.sass')[9] > (stat 'sgemm_pre_128.sass')[9])
{
    print `maxas.pl -p sgemm128.sass sgemm_pre_128.sass`;
    exit if $?;
    print `maxas.pl -i sgemm128.sass sgemm.cubin`;
    exit if $?;
    print `maxas.pl -e -k sgemm_kernel_128 sgemm.cubin sgemm_final_128.sass`;
}
if (!-f 'sgemm_pre_64.sass' || (stat 'sgemm64.sass')[9] > (stat 'sgemm_pre_64.sass')[9])
{
    print `maxas.pl -p sgemm64.sass sgemm_pre_64.sass`;
    exit if $?;
    print `maxas.pl -i sgemm64.sass sgemm.cubin`;
    exit if $?;
    print `maxas.pl -e -k sgemm_kernel_64 sgemm.cubin sgemm_final_64.sass`;
}

#print `Release\\sgemm.exe $_ 20` foreach (80,60,40,30,20,10,9,8,7,6,5,4,3,2);

`Release\\sgemm.exe 64 5 $CU_AD_FORMAT_FLOAT`;

print `Release\\sgemm.exe 64 20 $CU_AD_FORMAT_UNSIGNED_INT8`;
exit;

my %data;
foreach my $thread128 (4 .. 64)
{
    my $N = $thread128 * 128;

    my $iterations = int(20 * (64 * 128)**3 / $N**3);
    $iterations = 10000 if $iterations > 10000;

    print "$N $iterations\n";

    my $data = `Release\\sgemm.exe $thread128 $iterations $CU_AD_FORMAT_UNSIGNED_INT16`;

    foreach my $bench (split "\n", $data)
    {
        if ($bench =~ /^(\w+)\s+GFLOPS: ([0-9.]+) /)
        {
            push @{$data{$N}}, $2;
            print "$1 $2\n";
        }
    }
}
print join("\t", qw(size Max64 Max128 Cub64 Cub128)), "\n";

foreach my $N (sort { $a <=> $b } keys %data)
{
    print join("\t", @{$data{$N}}), "\n";
}


#print $data;

__END__


64 * 128 * 16 * 1.620 * .931 / 520

Max64  GFLOPS: 1377.38 (size: 256, iterations: 2000)
Max128 GFLOPS: 973.70 (size: 256, iterations: 2000)
Cub64  GFLOPS: 1272.42 (size: 256, iterations: 2000)
Cub128 GFLOPS: 948.15 (size: 256, iterations: 2000)

my @data = grep /\S/, split "\n", $data;

my $min;
my %smData;
my @sdata;
foreach (@data)
{
    next if /GFLOPS/;

    my ($sm, $clock, $by, $bx) = split /\s+/;

    $smData{$sm} = $clock if !$smData{$sm} || $clock < $smData{$sm};

    $min = $clock if !$min || $clock < $min;

    push @sdata, [$sm, $clock, $by, $bx];
}

foreach (@sdata)
{
    $_->[1] -= $smData{$_->[0]};
}

foreach (sort {$a->[1] <=> $b->[1] || $a->[0] <=> $b->[0]} @sdata)
{
    printf "%02d %8u  by: %2d bx: %2d\n", @$_;

}


