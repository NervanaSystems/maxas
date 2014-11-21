#!/usr/bin/perl
use strict;

print `maxas.pl -i shared_sts16.sass microbench.cubin`;

exit if $?;

print `Release\\microbench.exe i 1 64`;


__END__

