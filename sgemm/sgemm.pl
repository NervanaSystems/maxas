#!/usr/bin/perl
use strict;

print `maxas.pl -p sgemm.sass sgemm_preprocessed.sass`;

print `maxas.pl -p -r sgemm.sass sgemm_preprocessed_regmap.sass`;

print `maxas.pl -i sgemm.sass sgemm.cubin`;

print `maxas.pl -e sgemm.cubin sgemm_extracted.sass`;

print `Release\\sgemm.exe 40 1`;


__END__

