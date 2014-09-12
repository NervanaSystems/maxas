#!/usr/bin/perl
use strict;

my $blocks    = 10;
my $thread128 = 8;
my $loop      = 0x19000;

my $threads = $thread128 *  128;

my $fops = 2 * $loop * 512 * $blocks * $threads;

`maxas.pl -i throughput.sass microbench.cubin`;

my $data = `Release\\microbench.exe $blocks $thread128 1 $fops`;

my ($gflops) = $data =~ /GFLOPS: ([0-9\.]+)/ms;

print "$data";

__END__


open my $fh, '>params.txt';

my $params = join "\t", 0, 4, 8;

print $fh $params;
close $fh;

foreach my $r1 (0..15)
{
    foreach my $r2 (0..15)
    {
        next if $r1 == $r2;

        open my $fh, '>params.txt';

        my $params = join "\t", $r1, $r2;

        print $fh $params;
        close $fh;

        `maxas.pl -i throughput.sass microbench.cubin`;

        my $data = `Release\\microbench.exe $blocks $thread128 1 $fops`;

        my ($gflops) = $data =~ /GFLOPS: ([0-9\.]+)/ms;

        print "$params\t$gflops\n";
    }
}