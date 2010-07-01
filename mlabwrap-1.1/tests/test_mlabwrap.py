##############################################################################
################### test_mlabwrap: unittests for mlabwrap ####################
##############################################################################
##
## o authors: Alexander Schmolck, Vivek Rathod
## o created: 2003-07-00 00:00:00+00:00

import sys, os, re
import gc
from pdb import pm # for debugging test failures
try:
    import numpy
    from numpy.random import rand, randn
    toscalar = lambda a:a.item()
except ImportError:
    import Numeric as numpy
    from MLab import rand, randn
    toscalar = lambda a:a.toscalar()
from tempfile import mktemp
try: # python >= 2.3 has better mktemp
    from tempfile import mkstemp as _mkstemp
    mktemp = lambda *args,**kwargs: _mkstemp(*args, **kwargs)[1]
except ImportError: pass
degensym_proxy = lambda s, rex=re.compile(r'(PROXY_VAL)\d+'): rex.sub(r'\1',s)


import unittest
TestCase = unittest.TestCase
TestSuite = unittest.TestSuite
try:
    import awmstest
    TestCase = awmstest.PermeableTestCase2
    TestSuite = awmstest.RotatingTestSuite
except ImportError: pass

from awmstools import indexme, without
from mlabwrap import *
BUFSIZE=4096 # must be the same as in mlabraw.cpp

#XXX for testing in running session with existing mlab
## mlab
## mlab = MlabWrap()
mlab._dont_proxy['cell'] = True
WHO_AT_STARTUP = mlab.who()
mlab._dont_proxy['cell'] = False
# FIXME should do this differentlya
funnies = without(WHO_AT_STARTUP, ['HOME', 'V', 'WLVERBOSE', 'MLABRAW_ERROR_'])
if funnies:
    print >> sys.stderr, "Hmm, got some funny stuff in matlab env: %s" % funnies

#FIXME both below untested
def fitString(s, maxCol=79, newlineReplacement="\\n"):
    if newlineReplacement or isinstance(newlineReplacement, basestring):
        s = s.replace("\n", newlineReplacement)
    if maxCol is not None and len(s) > maxCol:
        s = "%s..." % s[:maxCol-3]
    return s
class NumericTestCase(TestCase):
    """Simple extensio to TestCase to handle array equality tests 'correctly'
       (i.e. work around rich comparisons). Since array repr's can also be
       very large, the printing of large reprs is controlled by
       ``maxReprLength`` (None to print everything) and
       ``reprNewlineReplacement`` (None not to replace newlines in the repr).
       """
    maxReprLength          = 30   #
    reprNewlineReplacement = "\\n"

    def _reallyEqual(self, first, second, testShape=True):
        #FIXME should this check for identical argument type, too?
        res = first == second
        # find out if are dealing with a sized object; looking for a __len__
        # attr does *NOT* work, because of #$@-C extension crap
        try:
            len(res)
        except TypeError:
            return res
        else:
            # HACK
            if len(first) == len(second) == 0:
                return `first` == `second` # deal with empty arrays
            res = ((not testShape or numpy.shape(first) == numpy.shape(second)) and
                   # it is necessary to exclude 0 element arrays, because

                   # identical zero-element arrays don't compare true (``and True`` normalizes)
                   (not len(first) and not len(second)
                    or bool(numpy.alltrue((numpy.ravel(first == second))))))
        return res
    def _smallRepr(self, *args):
        return tuple([fitString(repr(arg), maxCol=self.maxReprLength,
                                  newlineReplacement=self.reprNewlineReplacement)
                      for arg in args])
    def assertEqual(self, first, second, msg=None):
        if not self._reallyEqual(first, second):
            raise self.failureException, \
                  (msg or '%s != %s' % self._smallRepr(first, second))

    assertEqual = failUnlessEqual = assertEqual
    def assertNotEqual(self, first, second, msg=None):
        if self._reallyEqual(first, second):
            raise self.failureException, \
                  (msg or '%s == %s' % self._smallRepr(first, second))
    assertNotEquals = failIfEqual = assertNotEqual
    def assertAlmostEqual(self, first, second, places=7, msg=None):
        if not (numpy.shape(first) == numpy.shape(second) and \
                self._reallyEqual(numpy.around(second-first, places), 0, testShape=False)):
            raise self.failureException, \
                  (msg or '%s != %s within %s places' % self._smallRepr(first,second,places))
    assertAlmostEquals = failUnlessAlmostEqual = assertAlmostEqual
    def assertNotAlmostEqual(self, first, second, places=7, msg=None):
        if not (numpy.shape(first) == numpy.shape(second) and \
                not self._reallyEqual(numpy.around(second-first, places), 0, testShape=False)):
            raise self.failureException, \
                  (msg or '%s == %s within %s places' % self._smallRepr(first,second,places))
    failIfAlmostEqual =  assertNotAlmostEquals = assertNotAlmostEqual


def _canonicalMShape(a):
    """Matlab arrays are rank-less (rank is specified by indexing), so all
    arrays w/ trailing 1s in their shape are equivalent. This returns the
    canonical form for comparison purposes: no trailing 1s in the shape,
    unless the array would otherwise become a scalar or vector.
    """
    s = list(a.shape)
    if len(s) < 2: s.append(1)
    while s[2:] and s[-1] == 1: s.pop()
    return a.reshape(s)


class mlabwrapTC(NumericTestCase):
##     def assertEqual(self, first, second):
##         res = first == second
##         if len(res):
##             res = numpy.shape(first) == numpy.shape(second) and \
##                   bool(numpy.alltrue((numpy.ravel(a1 == a2))))
##         super(TestCase, self).assertEquals(res, True)

    def testBasic(self):
        """Test basic behavior."""
        array = numpy.array
        from random import randrange
        for isComplex in [False, True]:
            for i in range(30):
                #flat vector
                if i % 3:
                    nDims = 1
                    dims = randrange(1, 20)
                #2 - 6 dimensions.
                else:
                    nDims = randrange(2, 7)
                    dims  = [randrange(1, 7) for j in range(nDims)]
               
                a = numpy.random.random(dims)
                if isComplex: a = a + 1j*numpy.random.random(dims)
                a1 = a.copy()
                mlab._set('a', a)
                #### test simple get ####
                self.assertEqual(_canonicalMShape(a), mlab._get('a'))
                self.assertEqual(a,a1)

                ### test sliced arrays (stride handling test) ###
                b = a1.copy()
                for i in range(nDims):
                    z=0
                    while not z: z = randrange(-3,4)
                    b = b[::z]
                mlab._set('b',b)
                self.assertEqual(_canonicalMShape(b),mlab._get('b'))
                self.assertEqual(a1,a)
                ########## test for aliasing problems ##########
                if nDims > 1:
                    newA = mlab._get('a')
                    newA -= 1e4
                    if len(newA):
                        self.assertNotEqual(newA, mlab._get('a'))
                self.assertEqual(a,a1)
                mlab.clear('a')
                mlab.clear('b')
        # the tricky diversity of empty arrays
        mlab._set('a', [[]])
        self.assertEqual(mlab._get('a'), numpy.zeros((1, 0), 'd'))
        mlab._set('a', numpy.zeros((0,0)))
        self.assertEqual(mlab._get('a'), numpy.zeros((0, 0), 'd'))
        mlab._set('a', [])
        self.assertEqual(mlab._get('a'), numpy.zeros((0, 0), 'd'))
        # complex empty
        mlab._set('a', numpy.zeros((0,0), 'D'))
        self.assertEqual(mlab._get('a'), numpy.zeros((0, 0), 'd')) #XXX
        # 0d
        mlab._set('a', -2)
        self.assertEqual(mlab._get('a'), array([       [-2.]]))
        mlab._set('a', array(-2))
        self.assertEqual(mlab._get('a'), array([       [-2.]]))
        mlab.clear('a')
        # try basic error handling
        self.failUnlessRaises(MlabError, mlab._get, 'dontexist')
        self.failUnlessRaises(MlabError,mlab.round)
        assert toscalar(mlab.round(1.6)) == 2.0
        self.assertEqual(mlab.max([20,10],nout=2), (numpy.array([[20]]), array([[1]])))
        self.assertEqual(mlab.max([20,10]), numpy.array([[20]]))

    def testDoc(self):
        """Test that docstring extraction works OK."""
        mlab.who.__doc__.index('WHO lists the variables in the current workspace')
    def setUp(self):
        """Back up options."""
        self.backup = {}
        for opt in """\
        _array_cast
        _autosync_dirs
        _flatten_row_vecs
        _flatten_col_vecs
        _clear_call_args
        _session
        _proxies
        _proxy_count
        _mlabraw_can_convert
        _dont_proxy""".split():
           self.backup[opt] = mlab.__dict__[opt]
        mlab.addpath(os.path.dirname(__file__)) # XXX
        print "ADDPATHed", os.path.dirname(__file__)
    def tearDown(self):
        """Reset options."""
        mlab.__dict__.update(self.backup)
    def testCallArgs(self):
        mlab._dont_proxy['cell'] = True
        try:
            mlab._clear_call_args = False
            mlab.sin(1.23)
            assert mlab._get('arg0__', True) == 1.23
            mlab._clear_call_args = True
            mlab.sin(1.23)
            assert not 'arg0__' in mlab.who()
        finally:
            mlab._clear_call_args = True
            mlab._dont_proxy['cell'] = False
    def testXXXSubtler(self):
        """test more subtle stuff. This must come last, hence the XXX"""
        import os, cPickle
        array = numpy.array
        # simple strings:
        assert (mlab._do("''"), mlab._do("'foobar'")) == ('', 'foobar')
        self.assertEqual(mlab.sort(1), numpy.array([[1.]]))
        self.assertEqual(mlab.sort([3,1,2]), numpy.array([[1.], [2.], [3.]]))
        self.assertEqual(mlab.sort(numpy.array([3,1,2])), numpy.array([[1.], [2.], [3.]]))
        sct = mlab._do("struct('type',{'big','little'},'color','red','x',{3 4})")
        bct = mlab._do("struct('type',{'BIG','little'},'color','red')")
        self.assertEqual(sct[1].x, numpy.array([[4]]))
        self.assertEqual(sct[0].x, numpy.array([[3]]))
        #FIXME sct[:].x wouldn't work, but currently I'm not sure that's my fault
        sct[1].x  = 'New Value'
        assert sct[1].x == 'New Value'
        assert bct[0].type == 'BIG' and sct[0].type == 'big'
        mlab._set('foo', 1)
        assert mlab._get('foo') == numpy.array([1.])
        assert (mlab._do("{'A', 'b', {3,4, {5,6}}}") !=
                ['A', 'b', [array([[ 3.]]), array([[ 4.]]),
                            [array([[ 5.]]), array([[ 6.]])]]])
        mlab._dont_proxy['cell'] = True
        self.assertEquals(mlab._do("{'a', 'b', {3,4, {5,6}}}"),
                          ['a', 'b', [array([ 3.]), array([ 4.]),
                                      [array([ 5.]), array([ 6.])]]])
        mlab._dont_proxy['cell'] = False
        mlab.clear('foo')
        self.assertRaises(MlabError, mlab._get, 'foo')
        # XXX: note to self: ``format compact`` in startup.m will cause this
        # test to fail, but should otherwise be harmless.
        self.assertEquals(degensym_proxy(repr(sct)),
                          "<MlabObjectProxy of matlab-class: 'struct'; "
                          "internal name: 'PROXY_VAL__'; has parent: no>\n"
                          "1x2 struct array with fields:\n"
                          "    type\n    color\n    x\n\n")
        #FIXME: add tests for assigning and nesting proxies
        ## ensure proxies work OK as arguments
        self.assertEqual(mlab.size(sct), array([[1., 2.]]))
        self.assertEqual(mlab.size(sct, 1), array([[1]]))
        # test that exceptions on calls with proxy arguments don't result in
        # trouble
        self.assertRaises(MlabError, mlab.svd, sct)
        self.assertEqual(mlab.size(sct, [2]), array([[2]]))
        mlab._dont_proxy['cell'] = True
        gc.collect()
        assert map(degensym_proxy,without(mlab.who(), WHO_AT_STARTUP)) == (
            ['PROXY_VAL__', 'PROXY_VAL__'])
        # test pickling
        pickleFilename = mktemp()
        f = open(pickleFilename, 'wb')
        try:
            cPickle.dump({'sct': sct, 'bct': bct},f,1)
            f.close()
            f = open(pickleFilename, 'rb')
            namespace = cPickle.load(f)
            f.close()
        finally:
            os.remove(pickleFilename)
        gc.collect()
        assert len(mlab._proxies) == 4, "%d proxies!" % len(mlab._proxies)
        assert namespace['sct'][1].x == 'New Value'
        namespace['sct'][1].x = 'Even Newer Value'
        assert namespace['sct'][1].x ==  'Even Newer Value'
        assert sct[1].x == 'New Value'
        del sct
        del bct
        del namespace['sct']
        del namespace['bct']
        mlab._set('bar', '1234')
        x = []
        mlab._do("disp 'hallo'" ,nout=0, handle_out=x.append)
        assert x[0] == 'hallo\n'
        mlab._dont_proxy['cell'] = False
        self.assertRaises(ValueError, getattr, mlab, "buggy('ipython lookup')")
    def testSparseArrays(self):
        """Make sure sparse arrays work."""
        s = mlab.sparse(numpy.zeros([100,100]))
        self.assertEqual(mlab.full(s), numpy.zeros([100,100]))
        # FIXME: add these once we have multi-dimensional proxying
##         s = mlab.sparse(numpy.zeros([100,100]))
##         self.assertEqual(s[0,0], 0.0)
##         self.assertEqual(s[99,99], 0.0)
##         t = mlab.sparse(numpy.array([[1.,2,3],[0,0,0],[4,5,6]]))
##         self.assertEqual(t[0,0], 1.0)
##         self.assertEqual(t[1,1], 0)
##         self.assertEqual(t[2,2], 6)

    def testProxyIndexing(self):
        "indexing and co: time for some advanced proxied __getitem__ and __setitem__ etc.."
        p=mlab.proxyTest(mlab.struct('a', 1, 'b', '2'))
        p.c = [[4,5]]
        assert p.a == 1.0
        assert p.b == '2'
        assert list(p.c.flat) == [4,5]
        # test all combinations of 1D indexing
        sv = mlab.proxyTest(range(4))
        assert sv[0] == 0
        sv[0] = -33
        assert sv[0] == -33
        # test curly indexing; the proxyTest class in matlab arbitrarily uses
        # string conversion on ``{}`` indexing to have something to distinguish
        # it from "normal" ``()`` indexing
        sv._[0] = '0'
        assert sv._[0] == '0' == str(int(toscalar(sv[0])))
        assert sv["some'string\nwith\\funny\tstuff"] == (
            "you ()-indexed with the string <<some'string\nwith\\funny\tstuff>>")

        # FIXME this is something to potentially add, but that also raises issues
##         assert numpy.ndim(sv) == 2 # FIXME change that to 1?
##         assert numpy.shape(sv[:]) == (4,1)  # FIXME change that to 1?
        assert list(sv[:].flat) == range(4)
        # more complicated "open-ended" slices aren't supported (yet)
        self.assertEqual(sv[0:], sv[:])
        self.assertEqual(sv[:-1], sv[0:-1])
        self.assertEqual(sv[0:-1:1], sv[:-1])
        self.assertEqual(sv[-4:], sv[:])
        self.assertEqual(sv[-4:-3], sv[0:1])
        for b in [None] + range(-4,4):
            for e in  [None] + range(-4,4):
                for s in [None,1]:
                    assert list(sv[b:e:s].flat) == range(4)[b:e:s], (
                        "sv[b:e:s]: %s (b,e,s): %s" % (sv[b:e:s], (b,e,s)))


        sv[:-1] = -numpy.arange(3)
        assert list(sv[:].flat) ==  [-x for x in range(3)] + [3]
        sv[:] = numpy.arange(4)
        assert list(sv[:].flat) == range(4)
        sv[-2:] = numpy.arange(2)+10
        assert list(sv[:].flat) == [0,1,10,11]

        # FIXME math ops aren't yet implemented
        # sv *= 10
        # sv[1:3] *= 10 # FIXME
        # sv + 3

        # FIXME multi-D stuff isn't either
##         sm = mlab.proxyTest(arange(6).reshape(3,2))
##         assert sm.ndim == 2
##         assert sm.shape == (3,2)
##         assert len(sm) == 3
##         assert len(sm).T

##         p.sv = sv
##         assert p.sv is sv
##         assert p.sv[:]


    def testRawMlabraw(self):
        """A few explicit tests for mlabraw"""
        import mlabraw
        #print "test mlabraw"
        self.assertRaises(TypeError, mlabraw.put, 33, 'a',1)
        self.assertRaises(TypeError, mlabraw.get, object(), 'a')
        self.assertRaises(TypeError, mlabraw.eval, object(), '1')

        # -100 is picked kinda arbitrarily to account for internal "overhead";
        # I don't want to hardcode the exact value; users can assume 1000
        # chars is safe
        mlabraw.eval(mlab._session, '1'*(BUFSIZE-100))
        assert numpy.inf == mlabraw.get(mlab._session, 'ans');
        # test for buffer overflow detection
        self.assertRaises(Exception, mlabraw.eval, mlab._session, '1'*BUFSIZE)
        self.assertEqual(mlabraw.eval(mlab._session, r"fprintf('1\n')"),'1\n')
        try:
            self.assertEqual(mlabraw.eval(mlab._session, r"1"),'')
        finally:
            mlabraw.eval(mlab._session,'clear ans')
        #print "tested mlabraw"

    def testOrder(self):
        """Testing order flags cause no problems"""
        try: import numpy
        except ImportError: return
        fa=numpy.array([[1,2,3],[4,5,6]],order='F')
        self.assertEqual(mlab.conj(fa),fa)
        self.assertEqual([[2]],mlab.subsref(fa, mlab.struct('type', '()', 'subs',mlab._do('{{1,2}}'))))

suite = TestSuite(map(unittest.makeSuite,
                               (mlabwrapTC,
                                )))
unittest.TextTestRunner(verbosity=2).run(suite)

#FIXME strangely enough we can't test this in the function!
gc.collect()
mlab._dont_proxy['cell'] = True
# XXX got no idea where HOME comes from, not there under win
assert without(mlab.who(), WHO_AT_STARTUP) == ['bar'], "who is:%r" % mlab.who()
mlab.clear()
assert without(mlab.who(), ['MLABRAW_ERROR_']) == [] == mlab._do('{}'),(
    "who is:%r" % mlab.who())
mlab._dont_proxy['cell'] = False
