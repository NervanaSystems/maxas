#!/usr/bin/perl
use strict;

print `maxas.pl -i xmad2.sass microbench.cubin`;

exit if $?;

print `./microbench i 1 128`;


__END__

