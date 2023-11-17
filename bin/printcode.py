from __future__ import print_function

import sys
import math

value = sys.argv[1]
value = int(value, 0)

values = [0,0,0]
values[0] = value & 0x000000000001ffff
values[1] = (value & 0x0000003fffe00000) >> 21
values[2] = (value & 0x07fffc0000000000) >> 42

for value in values:
    stall = value & 0xf
    thisyield = (value & 0x10) >> 4
    wrtdb = (value & 0x000e0) >> 5
    readb = (value & 0x00700) >> 8
    watdb = (value & 0x1f800) >> 11
    print('stall', stall, 'thisyield', thisyield, 'write', wrtdb, 'read',
      readb, 'watdb', watdb)
    print('%s:%s:%s:%s:%s' % (watdb, readb,wrtdb, thisyield, stall))

