#!/usr/bin/perl
use strict;

print `maxas.pl -p sgemm.sass sgemm_preprocessed.sass`;

print `maxas.pl -p -r sgemm.sass sgemm_preprocessed_regmap.sass`;

print `maxas.pl -i sgemm.sass sgemm.cubin`;

print `maxas.pl -e sgemm.cubin sgemm_extracted.sass`;

#print `Release\\sgemm.exe $_ 5` foreach (40,30,20,10,9,8,7,6,5,4,3,2,1);

print `Release\\sgemm.exe 40 1`

__END__

