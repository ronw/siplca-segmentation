/* -*- c-basic-offset: 2 -*-
  mlabraw -- A Python module that exposes the MATLAB(TM) engine
  interface and supports passing Numeric arrays back and forth. It is
  derived from Andrew Sterian's pymat.

  - TODO:
     - rank > 2 arrays
     - string arrays
     - arrays of different types (uint8 etc.)
     - cells and struct support
     - parameters that control autocasting

  - FIXME:
    * add test cases (different array types and 0d arrays and '__array__'able
      types)

  - Notes:
    * performance is bad, with all likelihood mostly because of the poor
      quality of Matlab's C-interface. On a machine on which a python function
      call costs about 2 us, a mlabraw action (I timed put and eval; the cost
      of evaling doubles for the try/catch variant of eval compared to
      oldeval; NULL engOutputBuffer has no effect) will cost something like 2
      ms (1000x increase). The joint overhead added by mlabwrap and
      round-tripping seems to be about 10x, so the total performance
      degradation compared to calling a C-function on a scalar can be 4-6
      orders of magnitude (of course for expensive computations such as
      inverses on large matrices even this doesn't matter). Interestingly
      copying arrays is also really slow:

        oc:~/mlabwrap-1.0b/> python -m timeit -n 10 -s 'from numpy import arange, reshape; from mlabraw import open, put, eval; s=open(); a=reshape(arange(100000),(-1,));' "put(s,'a',a)"
        10 loops, best of 3: 19.6 msec per loop
        oc:~/mlabwrap-1.0b/> python -m timeit -n 10 -s 'from numpy import arange, reshape; from mlabraw import open, put, eval; s=open(); a=reshape(arange(100000),(-1,));' "b=a.copy()"
        10 loops, best of 3: 1.45 msec per loop

      Before trying to optimize, it's presumably worth while attempting a move
      to ctypes and see whether the ctypes-inherent overhead matters at all.

  Revision History
  ================

  mlabraw revision 1.1 -- 2009-09-14 Vivek Rathod & Alexander Schmolck
  ----------------------------------------------------------------------------
  - Vivek Rathod implemented n-d array support (this also marks the
    definite end of all Numeric suppport).

  mlabraw revision 1.0.1 -- 2009-03-19 Alexander Schmolck (a.schmolck@gmx.net)
  ----------------------------------------------------------------------------
  * Bugfixes: - changed BUFSIZE based on empirical observations and added overflow check
              - check for sparse arrays

  mlabraw version 1.0 -- 2007-02-27 Alexander Schmolck (a.schmolck@gmx.net)
  ----------------------------------------------------------------------------
  A modified, bugfixed and renamed version of pymat.

   * Interface changes:
     - works with both numpy and Numeric (Numeric support will likely
       disappear in the future)
     - removed (buggy and conceptually dubious) autoconversion of matlab(tm)
       1xN or Nx1 matrices to Numeric flat vectors 1x.
     - added proper error reporting: if something goes wrong during a matlab
       Execution, now a `mlabraw.error`, is raised (not `RuntimeError`) with an
       appropriate error message (from matlab(tm), if applicable) is raised
       (rather a kludge, thanks to matlab(tm)'s braindead C-interface). Also,
       passing incorrect types to the functions of this module now raises
       TypeErrors. Bizzarre violations (out of matlab(tm)-memory) continue to
       raise RuntimeErrors.
     - multidimensional string arrays are no longer concatenated (but
       currently just ignored)

   * Bug fixes:
     - added 64bit support
     - fixed serious memory violation bug: conversion of all flat Numeric
       vectors caused illegal memory access (in the copyNumeric... routines).
     - fixed serious memory leak (objects copied into matlab space are never destroyed)
     - fixed other segfaults that resulted from passing 'wrong' argument types
       to `put` (0-d arrays (now converted), numbers (now converted) and other
       non-array types (now should cause a warning message))
     - removed broken autoconversion (see above)
     - fixed compatibility for matlab 6.5 (untested)

   * Misc:
     - reformated source code, kicked out NEQ and EQ and changed 0 to NULL
       where appropriate
     - reformated and adapted documentation.


  pymat version 1.0 -- December 26, 1998, Andrew Sterian (asterian@umich.edu)
  ---------------------------------------------------------------------------
   * Initial release

  Copyright & Disclaimer
  ======================
  Copyright (c) 2002-2009 Alexander Schmolck and Vivek Rathod

  Copyright (c) 2002-2009 Alexander Schmolck <a.schmolck@gmx.net>

  Copyright (c) 1998,1999 Andrew Sterian. All Rights Reserved. mailto: steriana@gvsu.edu

  Copyright (c) 1998,1999 THE REGENTS OF THE UNIVERSITY OF MICHIGAN. ALL RIGHTS RESERVED

  Permission to use, copy, modify, and distribute this software and its
  documentation for any purpose and without fee is hereby granted, provided
  that the above copyright notices appear in all copies and that both these
  copyright notices and this permission notice appear in supporting
  documentation, and that the name of The University of Michigan not be used
  in advertising or publicity pertaining to distribution of the software
  without specific, written prior permission.

  THIS SOFTWARE IS PROVIDED AS IS, WITHOUT REPRESENTATION AS TO ITS FITNESS
  FOR ANY PURPOSE, AND WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS OR
  IMPLIED, INCLUDING WITHOUT LIMITATION THE IMPLIED WARRANTIES OF
  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE REGENTS OF THE
  UNIVERSITY OF MICHIGAN SHALL NOT BE LIABLE FOR ANY DAMAGES, INCLUDING
  SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, WITH RESPECT TO ANY
  CLAIM ARISING OUT OF OR IN CONNECTION WITH THE USE OF THE SOFTWARE, EVEN IF
  IT HAS BEEN OR IS HEREAFTER ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

*/
#include <Python.h> // !!! must come before standard includes
#include <stdarg.h>
#include <cstdio>
#define MLABRAW_VERSION "1.0.1"
// We're not building a MEX file, we're building a standalone app.
#undef MATLAB_MEX_FILE

#ifdef WIN32
#include <windows.h>

// FIXME not yet tested under windows
#ifndef vsnprintf
#define vsnprintf _vsnprintf
#endif

#endif

#include <numpy/arrayobject.h>
# ifndef PyArray_SBYTE
#  include <numpy/oldnumeric.h>
#  include <numpy/old_defines.h>
# endif

#include <engine.h>
#include <matrix.h>
#ifndef _V7_3_OR_LATER
#define mwSize int
#define mwIndex int
#endif

#include<iostream>

#ifndef max
#define max(x,y) ((x) > (y) ? (x) : (y))
#define min(x,y) ((x) < (y) ? (x) : (y))
#endif

static inline mxArray* _getMatlabVar(PyObject *lHandle, char *lName){
#ifdef _V6_5_OR_LATER
  return engGetVariable((Engine *)PyCObject_AsVoidPtr(lHandle), lName);
#else
  return engGetArray((Engine *)PyCObject_AsVoidPtr(lHandle), lName);
#endif
}

static PyObject *mlabraw_error;

#define pyassert(x,y) if (! (x)) { _pyassert(y); goto error_return; }

static void _pyassert(const char *pStr)
{
  PyErr_SetString(PyExc_RuntimeError, pStr);
}

// XXX AFAIK there's no good portable way to *printf safely in C; this is a lame
// attempt
static inline bool my_snprintf(char *dst, size_t size, const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	int n = vsnprintf(dst, size, fmt, args);
	va_end(args);
	return n >- 1 && static_cast<size_t>(n) < size;
}

// FIXME: add string array support
static PyStringObject *mx2char(const mxArray *pArray)
{
  size_t buflen;
  char *buf;
  PyStringObject *lRetval;
  if (mxGetM(pArray) > 1) {
    PyErr_SetString(mlabraw_error, "Only 1 Dimensional strings are currently supported");
    return NULL;
  }
  buflen = mxGetN(pArray) + 1;
  buf = (char *)mxCalloc(buflen, sizeof(char));
  pyassert(buf, "Out of MATLAB(TM) memory");

  if (mxGetString(pArray, buf, buflen)) {
    PyErr_SetString(mlabraw_error, "Unable to extract MATLAB(TM) string");
    mxFree(buf);
    return NULL;
  }

  lRetval = (PyStringObject *)PyString_FromString(buf);
  mxFree(buf);
  return lRetval;
	error_return: return NULL;
}


static PyArrayObject *mx2numeric(const mxArray *pArray)
{
  //current function returns PyArrayObject in c order currently
  mwSize nd;
  npy_intp  pydims[NPY_MAXDIMS];
  PyArrayObject *lRetval = NULL,*t=NULL;
  const double *lPR;
  const double *lPI;
  pyassert(PyArray_API,
           "Unable to perform this function without NumPy installed");

  nd = mxGetNumberOfDimensions(pArray);
  {
    const mwSize *dims;
    dims = mxGetDimensions(pArray);
    for (mwSize i=0; i != nd; i++){
        pydims[i] = static_cast<npy_intp>(dims[i]);
    }
  }
 //this function creates a fortran array
  t = (PyArrayObject *)
    PyArray_New(&PyArray_Type,static_cast<npy_intp>(nd), pydims,
                mxIsComplex(pArray) ? PyArray_CDOUBLE : PyArray_DOUBLE,
                NULL, // strides
                NULL, // data
                0,    //(ignored itemsize),
                NPY_F_CONTIGUOUS, 
                NULL); //  obj
  if (t == NULL) return NULL;
  
  lPR  = mxGetPr(pArray);
  if (mxIsComplex(pArray)) {
    double *lDst = (double *)PyArray_DATA(t);
    // AWMS unsigned int almost certainly can overflow on some platforms!
    npy_intp numberOfElements = PyArray_SIZE(t);
    lPI = mxGetPi(pArray);
    for (mwIndex i = 0; i != numberOfElements; i++) {
      *lDst++ = *lPR++;
      *lDst++ = *lPI++;
    }
  }
  else {
    double *lDst = (double *)PyArray_DATA(t);
    npy_intp numberOfElements = PyArray_SIZE(t);
    for (mwIndex i = 0; i != numberOfElements; i++) {
      *lDst++ = *lPR++;
    }
  }
  
  lRetval = (PyArrayObject *)PyArray_FromArray(t,NULL,NPY_C_CONTIGUOUS|NPY_ALIGNED|NPY_WRITEABLE);
  Py_DECREF(t);
  
  return lRetval;
  error_return:
  return NULL;
}

//FIXME check complex case
template <class T>
static inline void copyNumericVector2Mx(T *pSrc, npy_intp pRows, double *pDst, npy_intp *pStrides)
{
  // this is a horrible HACK for 0-D arrays (which have no strides);
  // it should also work for shape (1,) 1D arrays.
  // XXX: check that 1Ds are always OK!
  if (pRows == 1){
    *pDst = *pSrc;
  }
  else {
    npy_intp lRowDelta = pStrides[0]/sizeof(T);
    for (npy_intp lRow=0; lRow != pRows; lRow++, pSrc += lRowDelta) {
      *pDst++ = *pSrc;
    }
  }
}

template <class T>
static inline void copyNumeric2Mx(T *p,int size,double * pRData)
{
  while(size --){
    *pRData++ = *p++;
  }
}
template <class T>
static inline void copyCplxNumericVector2Mx(T *pSrc, npy_intp pRows, double *pRData,
                              double *pIData, npy_intp *pStrides)
{
  npy_intp lRowDelta = pStrides[0]/sizeof(T);
  for (npy_intp lRow=0; lRow != pRows; lRow++, pSrc += lRowDelta) {
    *pRData++ = pSrc[0];
    *pIData++ = pSrc[1];
  }
}

template <class T>
static inline void copyCplxNumeric2Mx(T *p,int size,double *pRData,double *pIData)
{
    while(size--){
      *pRData++ = *p++;
      *pIData++ = *p++;
    }
}

static mxArray *makeMxFromNumeric(const PyArrayObject *pSrc)
{
  npy_intp lRows=0, lCols=0;
  bool lIsComplex;
  bool lIsNotAMatrix = false;
  double *lR = NULL;
  double *lI = NULL;
  mxArray *lRetval = NULL;
  mwSize dims[NPY_MAXDIMS];
  mwSize nDims = pSrc->nd;
  const PyArrayObject *ap=NULL;

  switch (pSrc->nd) {
  case 0:                       // XXX the evil 0D
    lRows = 1;
    lCols = 1;
    lIsNotAMatrix = true;
    break;
  case 1:
    lRows = pSrc->dimensions[0];
    lCols = min(1, lRows); // for array([]): to avoid zeros((0,1)) !
    lIsNotAMatrix = true;
    break;
  default:
      for (mwSize i = 0;i != nDims; i++) {
        dims[i]=(mwSize)pSrc->dimensions[i];
      }
    break;
  }
  switch (pSrc->descr->type_num) {
  case PyArray_OBJECT:
    PyErr_SetString(PyExc_TypeError, "Non-numeric array types not supported");
    return NULL;
  case PyArray_CFLOAT:
  case PyArray_CDOUBLE:
    lIsComplex = true;
    break;
  default:
    lIsComplex = false;
  }

  // converts to fortran order if not already
  if(!PyArray_ISFORTRAN(pSrc)){ 
    ap = (PyArrayObject * const)PyArray_FromArray((PyArrayObject*)pSrc,NULL,NPY_ALIGNED|NPY_F_CONTIGUOUS);
  }
  else{
    ap = pSrc;
  }

  if(lIsNotAMatrix)
    lRetval = mxCreateDoubleMatrix(lRows, lCols, lIsComplex ? mxCOMPLEX : mxREAL);
  else
    lRetval = mxCreateNumericArray(nDims,dims,mxDOUBLE_CLASS,lIsComplex ? mxCOMPLEX : mxREAL);

  if (lRetval == NULL) return NULL;
  lR = mxGetPr(lRetval);
  lI = mxGetPi(lRetval);
  if (lIsNotAMatrix) {
    void *p = PyArray_DATA(ap);
    switch (ap->descr->type_num) {
    case PyArray_CHAR:
      copyNumericVector2Mx((char *)(p), lRows, lR, pSrc->strides);
      break;

    case PyArray_UBYTE:
      copyNumericVector2Mx((unsigned char *)(p), lRows, lR, pSrc->strides);
      break;

    case PyArray_SBYTE:
      copyNumericVector2Mx((signed char *)(p), lRows, lR, pSrc->strides);
      break;

    case PyArray_SHORT:
      copyNumericVector2Mx((short *)(p), lRows, lR, pSrc->strides);
      break;

    case PyArray_INT:
      copyNumericVector2Mx((int *)(p), lRows, lR, pSrc->strides);
      break;

    case PyArray_LONG:
      copyNumericVector2Mx((long *)(p), lRows, lR, pSrc->strides);
      break;

    case PyArray_FLOAT:
      copyNumericVector2Mx((float *)(p), lRows, lR, pSrc->strides);
      break;

    case PyArray_DOUBLE:
      copyNumericVector2Mx((double *)(p), lRows, lR, pSrc->strides);
      break;

    case PyArray_CFLOAT:
      copyCplxNumericVector2Mx((float *)(p), lRows, lR, lI, pSrc->strides);
      break;

    case PyArray_CDOUBLE:
      copyCplxNumericVector2Mx((double *)(p), lRows, lR, lI, pSrc->strides);
      break;
    }
  } else {
    void *p = PyArray_DATA(ap);
    npy_intp size = PyArray_SIZE(pSrc);

    switch (pSrc->descr->type_num) {
    case PyArray_CHAR:
      copyNumeric2Mx((char *)p,size,lR);
      break;

    case PyArray_UBYTE:
      copyNumeric2Mx((unsigned char *)p,size,lR);
      break;

    case PyArray_SBYTE:
      copyNumeric2Mx((signed char *)p,size,lR);
      break;

    case PyArray_SHORT:
      copyNumeric2Mx((short *)p,size,lR);
      break;

    case PyArray_INT:
      copyNumeric2Mx((int *)p,size,lR);
      break;

    case PyArray_LONG:
      copyNumeric2Mx((long *)p,size,lR);
      break;

    case PyArray_FLOAT:
      copyNumeric2Mx((float *)p,size,lR);
      break;

    case PyArray_DOUBLE:
      copyNumeric2Mx((double *)p,size,lR);
      break;

    case PyArray_CFLOAT:
      copyCplxNumeric2Mx((float *)p,size,lR,lI);
      break;

    case PyArray_CDOUBLE:
      copyCplxNumeric2Mx((double *)p,size,lR,lI);
      break;
    }
  }
  
  if(ap != pSrc){
    Py_DECREF(const_cast<PyArrayObject *>(ap));
  }
  return lRetval;
}

//AWMS: FIXME think about non-numeric sequences and whether we should return a cell array instead
static mxArray *makeMxFromSeq(const PyObject *pSrc)
{
  mxArray *lRetval = NULL;
  mwIndex i;
  mwSize lSize;

  PyArrayObject *lArray =
    (PyArrayObject *) PyArray_ContiguousFromObject(const_cast<PyObject *>(pSrc),
                                                   PyArray_CDOUBLE, 0, 0);
  if (lArray == NULL) return NULL;
  // AWMS: FIXME this is not a particularly intelligent way to set about
  // things
  // If all imaginary components are 0, this is not a complex array.
  lSize = PyArray_SIZE(lArray);
  // Get at first imaginary element
  const double *lPtr = (const double *)(lArray->data) + 1;
  for (i=0; i != lSize; i++, lPtr += 2) {
    if (*lPtr != 0.0) break;
  }
  if (i >= lSize) {
    PyArrayObject *lNew = (PyArrayObject *)PyArray_Cast(lArray, PyArray_DOUBLE);
    Py_DECREF(lArray);
    lArray = lNew;
  }

  lRetval = makeMxFromNumeric(lArray);
  Py_DECREF(lArray);

  return lRetval;
}

static mxArray *numeric2mx(PyObject *pSrc)
{
  mxArray *lDst = NULL;

  pyassert(PyArray_API, "Unable to perform this function without NumPy installed");
  if (PyArray_Check(pSrc)) {
    lDst = makeMxFromNumeric((const PyArrayObject *)pSrc);
  } else if (PySequence_Check(pSrc)) {
    lDst = makeMxFromSeq(pSrc);
  } else if (PyObject_HasAttrString(pSrc, "__array__")) {
    PyObject *arp;
    arp = PyObject_CallMethod(pSrc, "__array__", NULL);
    lDst = makeMxFromNumeric((const PyArrayObject *)arp);
    Py_DECREF(arp);             // FIXME check this is correct;
  }
    else if (PyInt_Check(pSrc) || PyLong_Check(pSrc) ||
             PyFloat_Check(pSrc) || PyComplex_Check(pSrc)){
    PyObject *t;
    t = Py_BuildValue("(O)", pSrc);
//     t = PyTuple_New(1);
//     PyTuple_SetItem(t, 0, pSrc);
    lDst = makeMxFromSeq(t);
    Py_DECREF(t); // FIXME check this
  } else {

  }
  return lDst;

 error_return:
  return NULL;
  }

static mxArray *char2mx(const PyObject *pSrc)
{
  mxArray *lDst = NULL;

  lDst = mxCreateString(PyString_AsString(const_cast<PyObject *>(pSrc)));
  if (lDst == NULL) {
    PyErr_SetString(mlabraw_error, "Unable to create MATLAB(TM) string");
    return NULL;
  }

  return lDst;
}

//////////////////////////////////////////////////////////////////////////////
static char open_doc[] =
#ifdef WIN32
"open() -> handle\n"
"\n"
"Opens a MATLAB(TM) engine session\n"
"This function returns a handle to a new MATLAB(TM) engine session.\n"
"For compatibility with the UNIX version of the Mlabraw module, this\n"
"function takes a single optional string parameter, but this parameter\n"
"is always ignored under Win32.\n"
#else
"open([str]) -> handle\n"
"\n"
"Opens a MATLAB(TM) engine session\n"
"This function returns a handle to a new MATLAB(TM) engine session.\n"
"The optional 'str' parameter determines how MATLAB(TM) is started.\n"
"If empty or not specified, the session is started by executing\n"
"the simple command 'matlab'. Other options include specifying\n"
"a host name or a verbatim string to use to invoke the MATLAB\n"
"program. See the `engOpen()` documentation in the MATLAB(TM) API\n"
"reference for more information.\n"
#endif
;

PyObject * mlabraw_open(PyObject *, PyObject *args)
{
  Engine *ep;
  char *lStr = "\0"; // "matlab -check_malloc";
  if (! PyArg_ParseTuple(args, "|s:open", &lStr)) return NULL;

#ifdef WIN32
  ep = engOpen(NULL);
#else
  ep = engOpen(lStr);
#endif
  if (ep == NULL) {
    PyErr_SetString(mlabraw_error, "Unable to start MATLAB(TM) engine");
    return NULL;
  }
  return PyCObject_FromVoidPtr(ep, NULL);
}


static char close_doc[] =
"close(handle)\n"
"\n"
"Closes MATLAB(TM) session\n"
"\n"
"This function closes the MATLAB(TM) session whose handle was returned\n"
"by a previous call to open().\n"
;

PyObject * mlabraw_close(PyObject *, PyObject *args)
{
  PyObject *lHandle;

  if (! PyArg_ParseTuple(args, "O:close", &lHandle)) return NULL;

  if (engClose((Engine *)PyCObject_AsVoidPtr(lHandle)) != 0) {
    PyErr_SetString(mlabraw_error, "Unable to close session");
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static char eval_doc[] =
"eval(handle, string) -> str\n"
"\n"
"Evaluates string in MATLAB(TM) session\n"
"This function evaluates the given string in the MATLAB(TM) session\n"
"associated with the handle. The handle is returned from a previous\n"
"call to open().\n"
"\n"
"The output of the command is returned as a string.\n"
"\n"
"If there is an error a `mlabraw.error` with the error description is raised.\n"
;

PyObject * mlabraw_eval(PyObject *, PyObject *args)
{
  //XXX how large should this be? used to be 10000, but looks like matlab
  // hangs when it gets bigger than ~9000 I'm not aware of some actual limit
  // being documented anywhere; I don't want to make it too small since the
  // high overhead of engine calls means that it can be potentially useful to
  // eval large chunks.
  const int  BUFSIZE=4096;
  char* fmt = "try, %s; MLABRAW_ERROR_=0; catch, MLABRAW_ERROR_=1; end;";
  char buffer[BUFSIZE];
  char cmd[BUFSIZE];
  char *lStr;
  char *retStr = buffer;
  PyObject *ret;
  PyObject *lHandle;
  if (! PyArg_ParseTuple(args, "Os:eval", &lHandle, &lStr)) return NULL;
  if (! PyCObject_Check(lHandle)) {
    PyErr_SetString(PyExc_TypeError, "Invalid object passed as mlabraw session handle");
	return NULL;
  }

  bool ok = my_snprintf(cmd, BUFSIZE, fmt, lStr);
  if (not ok) {
	  PyErr_SetString(mlabraw_error,
					  "String too long to evaluate.");
	  return NULL;
  }
  // std::cout << "DEBUG: CMD " << cmd << std::endl << std::flush;
  engOutputBuffer((Engine *)PyCObject_AsVoidPtr(lHandle), retStr, BUFSIZE-1);
  if (engEvalString((Engine *)PyCObject_AsVoidPtr(lHandle), cmd) != 0) {
    PyErr_SetString(mlabraw_error,
                   "Unable to evaluate string in MATLAB(TM) workspace");
    return NULL;
  }
  {
    mxArray *lArray = NULL;
    char buffer2[BUFSIZE];
    char *retStr2 = buffer2;
    bool __mlabraw_error;
    if (NULL == (lArray = _getMatlabVar(lHandle, "MLABRAW_ERROR_")) ) {
      PyErr_SetString(mlabraw_error,
                      "Something VERY BAD happened whilst trying to evaluate string "
                      "in MATLAB(TM) workspace.");
      return NULL;
    }
    __mlabraw_error = (bool)*mxGetPr(lArray);
    mxDestroyArray(lArray);
    if (__mlabraw_error) {
      engOutputBuffer((Engine *)PyCObject_AsVoidPtr(lHandle), retStr2, BUFSIZE-1);
      if (engEvalString((Engine *)PyCObject_AsVoidPtr(lHandle),
                        "disp(subsref(lasterror(),struct('type','.','subs','message')))") != 0) {
        PyErr_SetString(mlabraw_error, "THIS SHOULD NOT HAVE HAPPENED!!!");
        return NULL;
      }
      PyErr_SetString(mlabraw_error, retStr2 + ((strncmp(">> ", retStr2, 3) == 0) ?  3 : 0));
      return NULL;
    }
  }
  if (strncmp(">> ", retStr, 3) == 0) { retStr += 3; } //FIXME
  ret = (PyObject *)PyString_FromString(retStr);
  return ret;
}

PyObject * mlabraw_oldeval(PyObject *, PyObject *args)
{
  //XXX how large should this be?
  const int  BUFSIZE=10000;
  char buffer[BUFSIZE];
  char *lStr;
  char *retStr = buffer;
  PyObject *ret;
  PyObject *lHandle;

  if (! PyArg_ParseTuple(args, "Os:eval", &lHandle, &lStr)) return NULL;
  if (! PyCObject_Check(lHandle)) {
    PyErr_SetString(PyExc_TypeError, "Invalid object passed as mlabraw session handle");
    return NULL;
  }
  engOutputBuffer((Engine *)PyCObject_AsVoidPtr(lHandle), retStr, BUFSIZE-1);
  if (engEvalString((Engine *)PyCObject_AsVoidPtr(lHandle), lStr) != 0) {
    PyErr_SetString(mlabraw_error,
                   "Unable to evaluate string in MATLAB(TM) workspace");
    return NULL;
  }
  // skip the prompt if there is one
  if (strncmp(">> ", retStr, 3) == 0) {
    retStr += 3;
  }
  else {
    //XXX I think there is no prompt under windoze
//     printf("###DEBUG: matlab output doesn't start with \">> \"!\n"
//            "It starts with: '%s'\n"
//            "The command was: '%s'\n", retStr, lStr);
  }
  // "??? " is how an error message begins in matlab
  // obviously there is no proper way to test whether a command was
  // succesful... AAARGH
  if (strncmp("??? ", retStr, 4) == 0) {
    PyErr_SetString(mlabraw_error, retStr + 4); // skip "??? "
    return NULL;
  }
  ret = (PyObject *)PyString_FromString(retStr);
  return ret;
}

static char get_doc[] =
"get(handle, name) -> array\n"
"\n"
"Gets a matrix from the MATLAB(TM) session\n"
"\n"
"This function extracts the matrix with the given name from a MATLAB\n"
"session associated with the handle. The handle is the return value from\n"
"a previous call to open(). The name parameter must be a string describing\n"
"the name of a matrix in the MATLAB(TM) workspace. On Win32 platforms,\n"
"only double-precision floating point arrays (real or complex) are supported.\n"
"1-D character strings are supported on UNIX platforms.\n"
"\n"
"Only 2-dimensional arrays are currently supported. Cell\n"
"arrays, structure arrays, etc. are not yet supported.\n"
"\n"
"The return value is a NumPy array with the same shape and elements as the\n"
"MATLAB(TM) array.\n"
;
PyObject * mlabraw_get(PyObject *, PyObject *args)
{
  char *lName;
  PyObject *lHandle;
  mxArray *lArray = NULL;
  PyObject *lDest = NULL;

  if (! PyArg_ParseTuple(args, "Os:get", &lHandle, &lName)) return NULL;
  if (! PyCObject_Check(lHandle)) {
    PyErr_SetString(PyExc_TypeError, "Invalid object passed as mlabraw session handle");
    return NULL;
  }

  lArray = _getMatlabVar(lHandle, lName);
  if (lArray == NULL) {
    PyErr_SetString(mlabraw_error,
                   "Unable to get matrix from MATLAB(TM) workspace");
    return NULL;
  }

  if (mxIsChar(lArray)) {
    lDest = (PyObject *)mx2char(lArray);
  } else if (mxIsDouble(lArray) and not mxIsSparse(lArray)) {
    lDest = (PyObject *)mx2numeric(lArray);
  } else {                      // FIXME structs, cells and non-double arrays
    PyErr_SetString(PyExc_TypeError, "Only strings and non-sparse numeric arrays are supported.");
  }
  mxDestroyArray(lArray);
  return lDest;
}

static char put_doc[] =
"put(handle, name, array).\n"
"\n"
"Places a matrix into the MATLAB(TM) session.\n"
"This function places the given array into a MATLAB(TM) workspace under the\n"
"name given with the 'name' parameter (which must be a string). The handle\n"
"is a value previously obtained from a call to open().\n"
"\n"
"The 'array' parameter must be either a NumPy array, list, or tuple\n"
"containing numbers, or a number, or a string. The MATLAB(TM) \n"
"array will have the same shape and values, with the following\n"
"exceptions: the element type will always double or complex and the\n"
"array-rank will always be 2 (i.e. a matrix).\n"
"\n"
"A string parameter is converted to a MATLAB char-valued array.\n"
;
PyObject * mlabraw_put(PyObject *, PyObject *args)
{
  char *lName;
  PyObject *lHandle;
  PyObject *lSource;
  mxArray *lArray = NULL;
  //FIXME should make these objects const
  if (! PyArg_ParseTuple(args, "OsO:put", &lHandle, &lName, &lSource)) return NULL;
  if (! PyCObject_Check(lHandle)) {
    PyErr_SetString(PyExc_TypeError, "Invalid object passed as mlabraw session handle");
    return NULL;
  }
  Py_INCREF(lSource);

  if (PyString_Check(lSource)) {
    lArray = char2mx(lSource);
  } else {
    lArray = numeric2mx(lSource);
  }
  Py_DECREF(lSource);

  if (lArray == NULL) {
    return NULL;   // Above converter already set error message
  }


// for matlab version >= 6.5 (FIXME UNTESTED)
#ifdef _V6_5_OR_LATER
  if (engPutVariable((Engine *)PyCObject_AsVoidPtr(lHandle), lName, lArray) != 0) {
#else
  mxSetName(lArray, lName);
  if (engPutArray((Engine *)PyCObject_AsVoidPtr(lHandle), lArray) != 0) {
#endif
    PyErr_SetString(mlabraw_error,
                   "Unable to put matrix into MATLAB(TM) workspace");
    mxDestroyArray(lArray);
    return NULL;
  }
  mxDestroyArray(lArray);
  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef MlabrawMethods[] = {
  { "open",       mlabraw_open,       METH_VARARGS, open_doc },
  { "close",      mlabraw_close,      METH_VARARGS, close_doc },
  { "oldeval",    mlabraw_oldeval,    METH_VARARGS, ""       },
  { "eval",       mlabraw_eval,       METH_VARARGS, eval_doc },  //FIXME doc
  { "get",        mlabraw_get,        METH_VARARGS, get_doc },
  { "put",        mlabraw_put,        METH_VARARGS, put_doc },
  { NULL,         NULL,               0           , NULL}, // sentinel
};

PyMODINIT_FUNC initmlabraw(void)
{
  PyObject *module =
    Py_InitModule4("mlabraw",
      MlabrawMethods,
"Mlabraw -- Low-level MATLAB(tm) Engine Interface\n"
"\n"
"  open  - Open a MATLAB(tm) engine session\n"
"  close - Close a MATLAB(tm) engine session\n"
"  eval  - Evaluates a string in the MATLAB(tm) session\n"
"  get   - Gets a matrix from the MATLAB(tm) session\n"
"  put   - Places a matrix into the MATLAB(tm) session\n"
"\n"



"The Numeric package must be installed for this module to be used.\n"
"\n"
"Copyright & Disclaimer\n"
"======================\n"
"Copyright (c) 2002-2007 Alexander Schmolck <a.schmolck@gmx.net>\n"
"\n"
"Copyright (c) 1998,1999 Andrew Sterian. All Rights Reserved. mailto: steriana@gvsu.edu\n"
"\n"
"Copyright (c) 1998,1999 THE REGENTS OF THE UNIVERSITY OF MICHIGAN. ALL RIGHTS RESERVED \n"
"\n"
"Permission to use, copy, modify, and distribute this software and its\n"
"documentation for any purpose and without fee is hereby granted, provided\n"
"that the above copyright notices appear in all copies and that both these\n"
"copyright notices and this permission notice appear in supporting\n"
"documentation, and that the name of The University of Michigan not be used\n"
"in advertising or publicity pertaining to distribution of the software\n"
"without specific, written prior permission.\n"
"\n"
"THIS SOFTWARE IS PROVIDED AS IS, WITHOUT REPRESENTATION AS TO ITS FITNESS\n"
"FOR ANY PURPOSE, AND WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS OR\n"
"IMPLIED, INCLUDING WITHOUT LIMITATION THE IMPLIED WARRANTIES OF\n"
"MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE REGENTS OF THE\n"
"UNIVERSITY OF MICHIGAN SHALL NOT BE LIABLE FOR ANY DAMAGES, INCLUDING\n"
"SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, WITH RESPECT TO ANY\n"
"CLAIM ARISING OUT OF OR IN CONNECTION WITH THE USE OF THE SOFTWARE, EVEN IF\n"
"IT HAS BEEN OR IS HEREAFTER ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.\n"
"\n",
       0,
       PYTHON_API_VERSION);

  /* This macro, defined in arrayobject.h, loads the Numeric API interface */
  import_array();
  PyModule_AddStringConstant(module, "__version__", MLABRAW_VERSION);
  mlabraw_error = PyErr_NewException("mlabraw.error", NULL, NULL);
  Py_INCREF(mlabraw_error);
  PyModule_AddObject(module, "error", mlabraw_error);
}
