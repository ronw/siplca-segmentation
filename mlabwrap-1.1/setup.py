#!/usr/bin/env python
##############################################################################
########################### setup.py for mlabwrap ############################
##############################################################################
## o author: Alexander Schmolck (a.schmolck@gmx.net)
## o created: 2003-08-07 17:15:22+00:40
__version__ = '1.1'
####################################################################
##### VARIABLES YOU MIGHT HAVE TO CHANGE FOR YOUR INSTALLATION #####
##### (if setup.py fails to guess the right values for them)   #####
####################################################################
MATLAB_COMMAND = 'matlab'   # specify a full path if not in PATH
MATLAB_VERSION = None       # e.g: 6 (one of (6, 6.5, 7, 7.3))
                            #      7.3 includes later versions as well
MATLAB_DIR= None            # e.g: '/usr/local/matlab'; 'c:/matlab6'
PLATFORM_DIR=None           # e.g: 'glnx86'; r'win32/microsoft/msvc60'
EXTRA_COMPILE_ARGS=None     # e.g: ['-G']

# hopefully these 3 won't need modification
MATLAB_LIBRARIES=None       # e.g: ['eng', 'mx', 'mat', 'mi', 'ut']
USE_NUMERIC=None            # use obsolete Numeric instead of numpy?
PYTHON_INCLUDE_DIR=None     # where to find numpy/*.h

SUPPORT_MODULES= ["awmstools"] # set to [] if already
                               # installed
########################### WINDOWS ONLY ###########################
#  Option 1: Visual Studio
#  -----------------------
#  only needed for Windows Visual Studio (tm) build
#  (adjust if necessary if you use a different version/path of VC)
VC_DIR='C:/Program Files/Microsoft Visual Studio .NET 2003/vc7'
# NOTE: You'll also need to adjust PLATFORM_DIR accordingly
#
#  Option 2: Borland C++
#  ---------------------
#  uncomment (and adjust if necessary) the following lines to use Borland C++
#  instead of VC:
#
#VC_DIR=None
#PLATFORM_DIR="win32/borland/bc54"

####################################################################
### NO MODIFICATIONS SHOULD BE NECESSARY BEYOND THIS POINT       ###
####################################################################
# *******************************************************************

from distutils.core import setup, Extension
import os, os.path, glob
import sys
import re
from tempfile import mktemp
if sys.version_info < (2,2):
    print >> sys.stderr, "You need at least python 2.2"
    sys.exit(1)
try: # python >= 2.3 has better mktemp
    from tempfile import mkstemp as _mkstemp
    mktemp = lambda *args,**kwargs: _mkstemp(*args, **kwargs)[1]
except ImportError: pass
try: # subprocess is available from python >= 2.4
    from subprocess import call
except ImportError:
    def call(cmd, stdout=None):
        # ignore stdout redirection
        # could also escape via ``re.sub(r'[\"$*()?!{}[]]', r'\\\1', cmd[0])``
        return os.spawnvp(os.P_WAIT, cmd[0], cmd)


if PYTHON_INCLUDE_DIR is None and not USE_NUMERIC:
    try:
        import numpy
        PYTHON_INCLUDE_DIR = numpy.get_include()
    except ImportError:
        print >> sys.stderr, "Warning: numpy not found. Still using Numeric?"
        try:
            import Numeric
            if USE_NUMERIC is None: USE_NUMERIC=True
        except ImportError:
            print >> sys.stderr, "CANNOT FIND EITHER NUMPY *OR* NUMERIC"

def matlab_params(cmd, is_windows, extra_args):
    param_fname = mktemp()
    # XXX I have no idea why '\n' instead of the ``%c...,10`` hack fails - bug
    # in matlab's cmdline parsing? (``call`` doesn't do shell mangling, so that's not it...)
    code = ("fid = fopen('%s', 'wt');" % param_fname +
            r"fprintf(fid, '%s%c%s%c%s%c', version, 10, matlabroot, 10, computer, 10);" +
            "fclose(fid); quit")
    if is_windows:
        code = '"' + code + '"'
    cmd += ['-r', code]
    fh = None
    try:
        error = call(cmd, **extra_args)
        if error:
            sys.exit('''INSTALL ABORT: %r RETURNED ERROR CODE %d
PLEASE MAKE SURE matlab IS IN YOUR PATH!
''' % (" ".join(cmd), error))
        fh = open(param_fname)
        ver, pth, platform = iter(fh)
        return (float(re.match(r'\d+.\d+',ver).group()),
                pth.rstrip(), platform.rstrip().lower())
    finally:
        if fh: fh.close()
        try:
            os.remove(param_fname)
        except OSError, msg: # FIXME
            print >> sys.stderr, """
WINDOWS SPECIFIC ISSUE? Unable to remove %s; please delete it manually
%s
""" % (param_fname, msg)

# windows
WINDOWS=sys.platform.startswith('win')
if None in (MATLAB_VERSION, MATLAB_DIR, PLATFORM_DIR):
    cmd = [MATLAB_COMMAND, "-nodesktop",  "-nosplash"]
    if WINDOWS:
        extra_args = {}
        cmd += ["-wait"]
    else:
        extra_args = dict(stdout=open('/dev/null', 'wb'))   
    # FIXME: it is necessary to call matlab to figure out unspecified install
    # parameters but only if the user actually intends to build something
    # (e.g. not for making a sdist or running a clean or --author-email etc.).
    # Unfortunately I can't see a clean way to do that, so this nasty kludge
    # attempts to avoid calling matlab unless required.
    if len(sys.argv) > 1 and not (
        re.search("sdist|clean", sys.argv[1]) or
        len(sys.argv) == 2 and sys.argv[1].startswith('--') or
        sys.argv[-1].startswith('--help')):
        queried_version, queried_dir, queried_platform_dir = matlab_params(cmd, WINDOWS, extra_args)
    else:
        queried_version, queried_dir, queried_platform_dir = ["WHATEVER"]*3
    MATLAB_VERSION = MATLAB_VERSION or queried_version
    MATLAB_DIR = MATLAB_DIR or queried_dir
    PLATFORM_DIR = PLATFORM_DIR or queried_platform_dir
if WINDOWS:
    WINDOWS=True
    EXTENSION_NAME = 'mlabraw'
    MATLAB_LIBRARIES = MATLAB_LIBRARIES or 'libeng libmx'.split()
    CPP_LIBRARIES = [] #XXX shouldn't need CPP libs for windoze
# unices
else:
    EXTENSION_NAME = 'mlabrawmodule'
    if not MATLAB_LIBRARIES:
        if MATLAB_VERSION >= 6.5:
            MATLAB_LIBRARIES = 'eng mx mat ut'.split()
        else:
            MATLAB_LIBRARIES = 'eng mx mat mi ut'.split()
    CPP_LIBRARIES = ['stdc++'] #XXX strangely  only needed on some linuxes
    if sys.platform.startswith('sunos'):
        EXTRA_COMPILE_ARGS = EXTRA_COMPILE_ARGS or ['-G']



if MATLAB_VERSION >= 7 and not WINDOWS:
    MATLAB_LIBRARY_DIRS = [MATLAB_DIR + "/bin/" + PLATFORM_DIR]
else:
    MATLAB_LIBRARY_DIRS = [MATLAB_DIR + "/extern/lib/" + PLATFORM_DIR]
MATLAB_INCLUDE_DIRS = [MATLAB_DIR + "/extern/include"] #, "/usr/include"
if WINDOWS:
    if VC_DIR:
        MATLAB_LIBRARY_DIRS += [VC_DIR + "/lib"]
        MATLAB_INCLUDE_DIRS += [VC_DIR + "/include", VC_DIR + "/PlatformSDK/include"]
    else:
        print "Not using Visual C++; fiddling paths for Borland C++ compatibility"
        MATLAB_LIBRARY_DIRS = [mld.replace('/','\\') for mld in  MATLAB_LIBRARY_DIRS]
DEFINE_MACROS=[]
if MATLAB_VERSION >= 6.5:
    DEFINE_MACROS.append(('_V6_5_OR_LATER',1))
if MATLAB_VERSION >= 7.3:
    DEFINE_MACROS.append(('_V7_3_OR_LATER',1))
if USE_NUMERIC:
    DEFINE_MACROS.append(('MLABRAW_USE_NUMERIC', 1))
setup (# Distribution meta-data
       name = "mlabwrap",
       version = __version__,
       description = "A high-level bridge to matlab",
       author = "Alexander Schmolck",
       author_email = "A.Schmolck@gmx.net",
       py_modules = ["mlabwrap"] + SUPPORT_MODULES,
       url='http://mlabwrap.sourceforge.net',
       ext_modules = [
          Extension(EXTENSION_NAME, ['mlabraw.cpp'],
                    define_macros=DEFINE_MACROS,
                    library_dirs=MATLAB_LIBRARY_DIRS ,
                    runtime_library_dirs=MATLAB_LIBRARY_DIRS,
                    libraries=MATLAB_LIBRARIES + CPP_LIBRARIES,
                    include_dirs=MATLAB_INCLUDE_DIRS + (PYTHON_INCLUDE_DIR and [PYTHON_INCLUDE_DIR] or []),
                    extra_compile_args=EXTRA_COMPILE_ARGS,
                    ),
        ]
    )
