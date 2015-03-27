require 5.10.0;
use ExtUtils::MakeMaker;
# See lib/ExtUtils/MakeMaker.pm for details of how to influence
# the contents of the Makefile that is written.
WriteMakefile(
    NAME              => 'MaxAs::MaxAs',
    VERSION_FROM      => 'lib/MaxAs/MaxAs.pm', # finds $VERSION
    EXE_FILES         => ['bin/maxas.pl'],
    PREREQ_PM         => {Carp => 1.29, Data::Dumper => 2.145},
    LICENSE           => 'MIT',
    ($] >= 5.005 ?     ## Add these new keywords supported since 5.005
      (ABSTRACT_FROM  => 'lib/MaxAs/MaxAs.pm', # retrieve abstract from module
       AUTHOR         => 'Scott Gray <sgray@nervanasys.com>') : ()),
);
