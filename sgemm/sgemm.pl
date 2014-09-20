#!/usr/bin/perl
use strict;

print `maxas.pl -p sgemm128.sass sgemm_preprocessed.sass`;
#print `maxas.pl -p sgemm64.sass sgemm_preprocessed.sass`;

print `maxas.pl -i sgemm128.sass sgemm.cubin`;
print `maxas.pl -i sgemm64.sass sgemm.cubin`;

exit if $?;

print `maxas.pl -e -k sgemm_kernel_128 sgemm.cubin sgemm_final.sass`;
#print `maxas.pl -e -k sgemm_kernel_64 sgemm.cubin sgemm_final.sass`;


#print `Release\\sgemm.exe $_ 20` foreach (80,60,40,30,20,10,9,8,7,6,5,4,3,2);

print `Release\\sgemm.exe 80 1`

__END__

