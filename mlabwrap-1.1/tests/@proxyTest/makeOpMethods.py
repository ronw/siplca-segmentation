#!/bin/env python
# 2007-03-27 generate boilerplate matlab methods
from awmstools import spitOut
BINARY_ARITH = '''plus
minus
mtimes
times
mpower
power
mldivide
mrdivide
ldivide
rdivide

eq
ne
lt
gt
le
ge

horzcat
vertcat
'''.split()
UNARY_ARITH='''
uminus
uplus
transpose
ctranspose

'''.split() # XXX: size needs to be handcoded

for op in BINARY_ARITH:
    spitOut(file=op + ".m", s=
            '''function res=%(op)s(X,Y)
res=%(op)s(X.data, Y.data);
end
            ''' % dict(op=op))
for op in UNARY_ARITH:
    spitOut(file=op + ".m", s=
            '''function res=%(op)s(X)
res=%(op)s(X.data);
end
'''% dict(op=op))
