package Cubin;

use strict;
use Data::Dumper;

my @Elf32_Hdr = qw(
    H8  magic
    C   fileClass
    C   encoding
    C   fileVersion
    H18 padding
    S   type
    S   machine
    L   version
    L   entry
    L   phOffset
    L   shOffset
    L   flags
    S   ehSize
    S   phEntSize
    S   phNum
    S   shEntSize
    S   shNum
    S   shStrIndx
);
my @Elf64_Hdr = qw(
    H8  magic
    C   fileClass
    C   encoding
    C   fileVersion
    H18 padding
    S   type
    S   machine
    L   version
    Q   entry
    Q   phOffset
    Q   shOffset
    L   flags
    S   ehSize
    S   phEntSize
    S   phNum
    S   shEntSize
    S   shNum
    S   shStrIndx
);
my @Elf32_PrgHdr = qw(
    L   type
    L   offset
    L   vaddr
    L   paddr
    L   fileSize
    L   memSize
    L   flags
    L   align
);
my @Elf64_PrgHdr = qw(
    L   type
    L   flags
    Q   offset
    Q   vaddr
    Q   paddr
    Q   fileSize
    Q   memSize
    Q   align
);
my @Elf32_SecHdr = qw(
    L   name
    L   type
    L   flags
    L   addr
    L   offset
    L   size
    L   link
    L   info
    L   align
    L   entSize
);
my @Elf64_SecHdr = qw(
    L   name
    L   type
    Q   flags
    Q   addr
    Q   offset
    Q   size
    L   link
    L   info
    Q   align
    Q   entSize
);
my @Elf32_SymEnt = qw(
    L   name
    L   value
    L   size
    C   info
    C   other
    S   shIndx
);
my @Elf64_SymEnt = qw(
    L   name
    C   info
    C   other
    S   shIndx
    Q   value
    Q   size
);
my @symBind = qw(LOCAL GLOBAL WEAK);

# Split the Elf Header defs into template strings (T) and corresponding hash keys columns (C)
my (@elfHdrT, @prgHdrT, @secHdrT, @symHdrT, @elfHdrC, @prgHdrC, @secHdrC, @symHdrC);

$elfHdrT[1] = join '', grep { length($_) <= 3} @Elf32_Hdr;
$prgHdrT[1] = join '', grep { length($_) <= 3} @Elf32_PrgHdr;
$secHdrT[1] = join '', grep { length($_) <= 3} @Elf32_SecHdr;
$symHdrT[1] = join '', grep { length($_) <= 3} @Elf32_SymEnt;

$elfHdrT[2] = join '', grep { length($_) <= 3} @Elf64_Hdr;
$prgHdrT[2] = join '', grep { length($_) <= 3} @Elf64_PrgHdr;
$secHdrT[2] = join '', grep { length($_) <= 3} @Elf64_SecHdr;
$symHdrT[2] = join '', grep { length($_) <= 3} @Elf64_SymEnt;

$elfHdrC[1] = [ grep { length($_) > 3} @Elf32_Hdr    ];
$prgHdrC[1] = [ grep { length($_) > 3} @Elf32_PrgHdr ];
$secHdrC[1] = [ grep { length($_) > 3} @Elf32_SecHdr ];
$symHdrC[1] = [ grep { length($_) > 3} @Elf32_SymEnt ];

$elfHdrC[2] = [ grep { length($_) > 3} @Elf64_Hdr    ];
$prgHdrC[2] = [ grep { length($_) > 3} @Elf64_PrgHdr ];
$secHdrC[2] = [ grep { length($_) > 3} @Elf64_SecHdr ];
$symHdrC[2] = [ grep { length($_) > 3} @Elf64_SymEnt ];

# Load a cubin ELF file
sub new
{
    my ($package, $file) = @_;

    my $cubin = bless { fileName => $file }, $package;

    open my $fh, $file or die "$file: $!";
    binmode($fh);

    # Read in assuming 32 bit header
    my $data;
    read $fh, $data, 0x34;
    my $elfHdr = $cubin->{elfHdr} = {};
    @{$elfHdr}{@{$elfHdrC[1]}} = unpack $elfHdrT[1], $data;

    # 1: 32bit, 2: 64bit
    my $class = $elfHdr->{fileClass};

    # re-read in with 64 bit header if needed
    if ($class == 2)
    {
        seek $fh, 0, 0;
        read $fh, $data, 0x46;
        @{$elfHdr}{@{$elfHdrC[$class]}} = unpack $elfHdrT[$class], $data;
    }

    # verify sm_50 cubin
    $cubin->{Arch} = $elfHdr->{flags} & 0xFF;
    die "Cubin not in sm_50 or greater format. Found: sm_$cubin->{Arch}\n" if $cubin->{Arch} < 50;

    # Read in Program Headers
    seek $fh, $elfHdr->{phOffset}, 0;
    foreach (1 .. $elfHdr->{phNum})
    {
        read $fh, $data, $elfHdr->{phEntSize};

        my %prgHdr = (Indx => $_ - 1);
        @prgHdr{@{$prgHdrC[$class]}} = unpack $prgHdrT[$class], $data;
        push @{$cubin->{prgHdrs}}, \%prgHdr;
    }

    # Read in Section Headers
    seek $fh, $elfHdr->{shOffset}, 0;
    foreach (1 .. $elfHdr->{shNum})
    {
        read $fh, $data, $elfHdr->{shEntSize};

        my %secHdr = (Indx => $_ - 1);
        @secHdr{@{$secHdrC[$class]}} = unpack $secHdrT[$class], $data;
        push @{$cubin->{secHdrs}}, \%secHdr;
    }

    # Read in Section data
    foreach my $secHdr (@{$cubin->{secHdrs}})
    {
        $data = '';
        # Skip sections with no data (type NULL or NOBITS)
        if ($secHdr->{size} && $secHdr->{type} != 8)
        {
            seek $fh, $secHdr->{offset}, 0;
            read $fh, $data, $secHdr->{size};
        }
        # Convert string tables to maps
        if ($secHdr->{type} == 3) # STRTAB
        {
            my $strTab = $secHdr->{StrTab} = {};
            my $indx   = 0;
            foreach my $str (split "\0", $data)
            {
                $strTab->{$indx} = $str;
                $indx += 1 + length($str);
            }
        }
        # Read in Symbol data
        if ($secHdr->{type} == 2) # SYMTAB
        {
            my $offset = 0;
            while ($offset < $secHdr->{size})
            {
                my $symEnt = {};
                @{$symEnt}{@{$symHdrC[$class]}} = unpack $symHdrT[$class], substr($data, $offset, $secHdr->{entSize});
                $offset += $secHdr->{entSize};

                push @{$secHdr->{SymTab}}, $symEnt;
            }
        }
        # Cache raw data for further processing and writing
        $secHdr->{Data} = unpack 'H*', $data;
    }
    close $fh;

    # Update section headers with their names.  Map names directly to headers.
    my $shStrTab = $cubin->{secHdrs}[$elfHdr->{shStrIndx}]{StrTab};
    foreach my $secHdr (@{$cubin->{secHdrs}})
    {
        $secHdr->{Name} = $shStrTab->{$secHdr->{name}};
        $cubin->{$secHdr->{Name}} = $secHdr;
    }

    # Update symbols with their names
    # For the Global functions, extract kernel meta data
    # Populate the kernel hash
    my $strTab = $cubin->{'.strtab'}{StrTab};
    foreach my $symEnt (@{$cubin->{'.symtab'}{SymTab}})
    {
        $symEnt->{Name} = $strTab->{$symEnt->{name}};

        # Attach symbol to section
        my $secHdr = $cubin->{secHdrs}[$symEnt->{shIndx}];
        $secHdr->{SymbolEnt} = $symEnt;

        # Look for symbols tagged FUNC
        if (($symEnt->{info} & 0x0f) == 0x02)
        {
            # Create a hash of kernels for output
            my $kernelSec = $cubin->{Kernels}{$symEnt->{Name}} = $secHdr;

            # Extract local/global/weak binding info
            $kernelSec->{Linkage} = $symBind[($symEnt->{info} & 0xf0) >> 4];

            # Extract the kernel instructions
            $kernelSec->{KernelData} = [ unpack "Q*", pack "H*", $kernelSec->{Data} ];

            # Extract the max barrier resource identifier used and add 1. Should be 0-16.
            # If a register is used as a barrier resource id, then this value is the max of 16.
            $kernelSec->{BarCnt} = ($kernelSec->{flags} & 0x01f00000) >> 20;

            # Extract the number of allocated registers for this kernel.
            $kernelSec->{RegCnt} = ($kernelSec->{info} & 0xff000000) >> 24;

            # Extract the size of shared memory this kernel uses.
            my $sharedSec = $kernelSec->{SharedSec} = $cubin->{".nv.shared.$symEnt->{Name}"};
            $kernelSec->{SharedSize} = $sharedSec ? $sharedSec->{size} : 0;

            # Attach constant0 section
            $kernelSec->{ConstantSec} = $cubin->{".nv.constant0.$symEnt->{Name}"};

            # Extract the kernel parameter data.
            my $paramSec = $kernelSec->{ParamSec} = $cubin->{".nv.info.$symEnt->{Name}"};
            if ($paramSec)
            {
                # Extract raw param data
                my @data = unpack "L*", pack "H*", $paramSec->{Data}; #map { sprintf '0x%08x', $_ }
                #print Dumper(\@data);
                #exit();

                $paramSec->{ParamData} = \@data;

                # InsCnt is the number of non-control instructions of a kernel (not including final EXIT, BRA and NOP instuctions)
                #TODO: this logic is sometimes wrong.. but it turns out you don't need to modify this value to edit a kernel
                $kernelSec->{InsCnt}   = $data[$#data-2] / 8; # the value is stored as a size

                # Find the first param delimiter
                my $idx = 0;
                $idx++ while $idx < @data && $data[$idx] != 0x00080a04;

                my $first = $data[$idx+2] & 0xFFFF;
                my $size  = $data[$idx+2] >> 16;
                $idx += 4;

                my @params;
                while ($idx < @data && $data[$idx] == 0x000c1704)
                {
                    # Get the ordinal, offset, size and pointer alignment for each param
                    my $ord    = $data[$idx+2] & 0xFFFF;
                    my $offset = sprintf '0x%02x', $first + ($data[$idx+2] >> 16);
                    my $psize  = $data[$idx+3] >> 18;
                    my $align  = $data[$idx+3] & 0x400 ? 1 << ($data[$idx+3] & 0x3ff) : 0;
                    unshift @params, "$ord:$offset:$psize:$align";
                    $idx += 4;
                }
                $kernelSec->{Params} = \@params;
                $kernelSec->{ParamCnt} = scalar @params;
            }
        }
        # Note GLOBALs found in this cubin
        elsif (($symEnt->{info} & 0x10) == 0x10)
        {
            $cubin->{Symbols}{$symEnt->{Name}} = $symEnt;
        }
    }
    return $cubin;
}
sub arch
{
    return shift()->{Arch};
}
sub listKernels
{
    return shift()->{Kernels};
}
sub listSymbols
{
    return shift()->{Symbols};
}
sub getKernel
{
    my ($cubin, $kernel) = @_;
    return $cubin->{Kernels}{$kernel};
}

sub modifyKernel
{
    my ($cubin, %params) = @_;

    my $kernelSec = $params{Kernel};
    my $newReg    = $params{RegCnt};
    my $newBar    = $params{BarCnt};
    my $newCnt    = $params{InsCnt};
    my $newData   = $params{KernelData};
    my $newSize   = @$newData * 8;

    my $elfHdr = $cubin->{elfHdr};
    my $class  = $elfHdr->{fileClass};

    die "255 register max" if $newReg > 255;
    die "new kernel size must be multiple of 8 instructions (64 bytes)" if $newSize & 63;
    die "16 is max barrier count" if $newBar > 16;

    my $paramSec = $kernelSec->{ParamSec};
    my $kernelName = $kernelSec->{SymbolEnt}{Name};

    # update the kernel
    $kernelSec->{KernelData} = $newData;
    $kernelSec->{Data}       = unpack "H*", pack "Q*", @$newData;

    if ($newReg != $kernelSec->{RegCnt})
    {
        print "Modified $kernelName RegCnt: $kernelSec->{RegCnt} => $newReg\n";
        $kernelSec->{RegCnt} = $newReg;
        $kernelSec->{info}  &= ~0xff000000;
        $kernelSec->{info}  |= $newReg << 24;
    }
    if ($newBar != $kernelSec->{BarCnt})
    {
        print "Modified $kernelName BarCnt: $kernelSec->{BarCnt} => $newBar\n";
        $kernelSec->{BarCnt} = $newBar;
        $kernelSec->{flags} &= ~0x01f00000;
        $kernelSec->{flags} |=  $newBar << 20;
    }
    # This logic is sometimes wrong but it's not required to modify to get the kernel working
    #if ($newCnt != $kernelSec->{InsCnt})
    #{
    #    print "Modified $kernelName InsCnt: $kernelSec->{InsCnt} => $newCnt\n";
    #    $kernelSec->{InsCnt} = $newCnt;
    #    my $data = $paramSec->{ParamData};
    #    $data->[$#$data-2] = $newCnt * 8;
    #    $paramSec->{Data} = unpack "H*", pack "L*", @$data;
    #}
    if ($newSize != $kernelSec->{size})
    {
        print "Modified $kernelName Size: $kernelSec->{size} => $newSize\n";

        # update kernel section
        my $delta = $newSize - $kernelSec->{size};
        $kernelSec->{size} = $newSize;

        # update symtab section
        $kernelSec->{SymbolEnt}{size} = $newSize;
        my $symSection = $cubin->{'.symtab'};
        $symSection->{Data} = '';
        foreach my $symEnt (@{$symSection->{SymTab}})
        {
            $symSection->{Data} .= unpack "H*", pack $symHdrT[$class], @{$symEnt}{@{$symHdrC[$class]}};
        }

        # update elf header offsets
        $elfHdr->{phOffset} += $delta;
        $elfHdr->{shOffset} += $delta;

        # update section header offsets
        foreach my $secHdr (@{$cubin->{secHdrs}})
        {
            $secHdr->{offset} += $delta if $secHdr->{offset} > $kernelSec->{offset};
        }

        # update program header offsets and sizes
        foreach my $prgHdr (@{$cubin->{prgHdrs}})
        {
            if ($kernelSec->{offset} < $prgHdr->{offset})
            {
                $prgHdr->{offset} += $delta;
            }
            # also update the size of any header that contains this kernel
            elsif ($kernelSec->{offset} < $prgHdr->{offset} + $prgHdr->{fileSize})
            {
                $prgHdr->{fileSize} += $delta;
                $prgHdr->{memSize}  += $delta;
            }
        }
    }
}

# Write out the cubin after modifying it.
sub write
{
    my ($cubin, $file) = @_;

    open my $fh, ">$file" or die "Error: could not open $file for writing: $!";
    binmode($fh);

    my $elfHdr = $cubin->{elfHdr};
    my $class  = $elfHdr->{fileClass};

    # write elf header
    print $fh pack $elfHdrT[$class], @{$elfHdr}{@{$elfHdrC[$class]}};
    my $pos = $elfHdr->{ehSize};

    # write section data
    foreach my $secHdr (@{$cubin->{secHdrs}})
    {
        # Skip NULL and NOBITS data sections
        next if $secHdr->{size} == 0 || $secHdr->{type} == 8;

        # Add any needed padding between sections
        my $pad = $pos % $secHdr->{align};
        if ($pad > 0)
        {
            $pad = $secHdr->{align} - $pad;
            print $fh join '', "\0" x $pad;
            $pos += $pad;
        }

        print $fh pack 'H*', $secHdr->{Data};
        $pos += $secHdr->{size};
    }

    # write section headers
    foreach my $secHdr (@{$cubin->{secHdrs}})
    {
        print $fh pack $secHdrT[$class], @{$secHdr}{@{$secHdrC[$class]}};
    }

    #write program headers
    foreach my $prgHdr (@{$cubin->{prgHdrs}})
    {
        print $fh pack $prgHdrT[$class], @{$prgHdr}{@{$prgHdrC[$class]}};
    }
    close $fh;
}

__END__

