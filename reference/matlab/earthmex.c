#include <stdarg.h>
#include <math.h>
#include "mex.h"
#include "earth.h"

// Fred Nicolls, May 2013
// Source for Matlab wrapper to Steve Milborrow's earth package for Friedman's multivariate
// adaptive regression splines (MARS).

#define CHECK(st) if( (long int)(st)==(long int)0 ) { \
mexPrintf("Check failed (Matlab) in %s, line %i\n", __FILE__, __LINE__); \
mexErrMsgTxt("Aborting"); }

// Required for error handling in earth.c
void error(const char *args, ...)       // params like printf
{
  char s[1000];
  va_list p;
  va_start(p, args);
  vsprintf(s, args, p);
  va_end(p);
  mexPrintf("\nError: %s\n", s);
  mexErrMsgTxt("Aborting");
}

//[BestGcv,nTerms,BestSet,bx,Dirs,Cuts,Residuals,Betas] = train(x,y,nMaxDegree,nMaxTerms,Penalty,Thresh,nMinSpan,
//  Prune,nFastK,FastBeta,NewVarPenalty,LinPreds,UseBetaCache,Trace)
void train(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int argn;
  const mxArray *mxx, *mxy;
  const double *Weights = NULL;
  int nCases, nPreds, nResp;
  int nMaxDegree, nMaxTerms;
  double Penalty, Thresh;
  int nMinSpan;
  bool Prune;
  int nFastK;
  double FastBeta, NewVarPenalty;
  const mxArray *mxLinPreds;  // nPreds x 1
  bool UseBetaCache;
  double Trace;
  
  // Input parameters
  argn = 0;
  CHECK( argn<nrhs );  mxx = prhs[argn++];
  CHECK( argn<nrhs );  mxy = prhs[argn++];
  CHECK( argn<nrhs );  nMaxDegree = (int)mxGetScalar( prhs[argn++] );
  CHECK( argn<nrhs );  nMaxTerms = (int)mxGetScalar( prhs[argn++] );
  CHECK( argn<nrhs );  Penalty = (double)mxGetScalar( prhs[argn++] );
  CHECK( argn<nrhs );  Thresh = (double)mxGetScalar( prhs[argn++] );
  CHECK( argn<nrhs );  nMinSpan = (int)mxGetScalar( prhs[argn++] );
  CHECK( argn<nrhs );  Prune = (int)mxGetScalar( prhs[argn++] )==0 ? false : true;
  CHECK( argn<nrhs );  nFastK = (int)mxGetScalar( prhs[argn++] );
  CHECK( argn<nrhs );  FastBeta = (double)mxGetScalar( prhs[argn++] );
  CHECK( argn<nrhs );  NewVarPenalty = (double)mxGetScalar( prhs[argn++] );
  CHECK( argn<nrhs );  mxLinPreds = prhs[argn++];
  CHECK( argn<nrhs );  UseBetaCache = (int)mxGetScalar( prhs[argn++] )==0 ? false : true;
  CHECK( argn<nrhs );  Trace = (double)mxGetScalar( prhs[argn++] );
  
  // Get input info and do checks
  CHECK( sizeof(int)==4 );
  CHECK( mxGetClassID(mxx)==mxDOUBLE_CLASS );
  CHECK( mxGetClassID(mxy)==mxDOUBLE_CLASS );
  nCases = (int)mxGetM(mxx);  nPreds = (int)mxGetN(mxx);  nResp = (int)mxGetN(mxy);
  CHECK( mxGetM(mxy)==nCases );
  CHECK( mxGetClassID(mxLinPreds)==mxINT32_CLASS );
  CHECK( mxGetM(mxLinPreds)==nPreds && mxGetN(mxLinPreds)==1 );
  
  // Create return quantities
  mxArray *mxBestGcv = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
  mxArray *mxnTerms = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
  bool *BestSet = (bool *)mxMalloc(nMaxTerms*sizeof(bool));  // no native bool type
  mxArray *mxbx = mxCreateNumericMatrix(nCases, nMaxTerms, mxDOUBLE_CLASS, mxREAL);
  mxArray *mxDirs = mxCreateNumericMatrix(nMaxTerms, nPreds, mxINT32_CLASS, mxREAL);
  mxArray *mxCuts = mxCreateNumericMatrix(nMaxTerms, nPreds, mxDOUBLE_CLASS, mxREAL);
  mxArray *mxResiduals = mxCreateNumericMatrix(nCases, nResp, mxDOUBLE_CLASS, mxREAL);
  mxArray *mxBetas = mxCreateNumericMatrix(nMaxTerms, nResp, mxDOUBLE_CLASS, mxREAL);
    
  // Function call for training
  Earth((double *)mxGetData(mxBestGcv), (int *)mxGetData(mxnTerms), BestSet, (double *)mxGetData(mxbx),
        (int *)mxGetData(mxDirs), (double *)mxGetData(mxCuts), (double *)mxGetData(mxResiduals),
        (double *)mxGetData(mxBetas), (double *)mxGetData(mxx), (double *)mxGetData(mxy), Weights,
        nCases, nResp, nPreds, nMaxDegree, nMaxTerms, Penalty, Thresh, nMinSpan, Prune,
        nFastK, FastBeta, NewVarPenalty, (int *)mxGetData(mxLinPreds), UseBetaCache, Trace, NULL);
  
  // Prepare outputs
  mxArray *mxBestSet = mxCreateNumericMatrix(nMaxTerms, 1, mxUINT8_CLASS, mxREAL);
  unsigned char *puc = (unsigned char *)mxGetData(mxBestSet);
  for( int i=0; i<nMaxTerms; i++ ) puc[i] = BestSet[i]==false ? 0 : 1;
  mxFree(BestSet);
  
  // Assign and return
  argn = 0;
  CHECK( argn<nlhs );  plhs[argn++] = mxBestGcv;
  CHECK( argn<nlhs );  plhs[argn++] = mxnTerms;
  CHECK( argn<nlhs );  plhs[argn++] = mxBestSet;
  CHECK( argn<nlhs );  plhs[argn++] = mxbx;
  CHECK( argn<nlhs );  plhs[argn++] = mxDirs;
  CHECK( argn<nlhs );  plhs[argn++] = mxCuts;
  CHECK( argn<nlhs );  plhs[argn++] = mxResiduals;
  CHECK( argn<nlhs );  plhs[argn++] = mxBetas;
  
  return;
}

//y = predict(x,BestSet,Dirs,Cuts,Betas,nTerms)
void predict(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int argn;
  const mxArray *mxx, *mxBestSet, *mxDirs, *mxCuts, *mxBetas;
  int nTerms;
  int nCases, nPreds, nResp, nMaxTerms;
  
  // Input parameters
  argn = 0;
  CHECK( argn<nrhs );  mxx = prhs[argn++];
  CHECK( argn<nrhs );  mxBestSet = prhs[argn++];
  CHECK( argn<nrhs );  mxDirs = prhs[argn++];
  CHECK( argn<nrhs );  mxCuts = prhs[argn++];
  CHECK( argn<nrhs );  mxBetas = prhs[argn++];
  CHECK( argn<nrhs );  nTerms = (int)mxGetScalar( prhs[argn++] );
  
  // Get input info and do checks
  CHECK( mxGetClassID(mxx)==mxDOUBLE_CLASS );
  nCases = (int)mxGetM(mxx);  nPreds = (int)mxGetN(mxx);
  CHECK( mxGetClassID(mxBestSet)==mxUINT8_CLASS );
  nMaxTerms = (int)mxGetM(mxBestSet);  CHECK( mxGetN(mxBestSet)==1 );
  CHECK( mxGetClassID(mxDirs)==mxINT32_CLASS );
  CHECK( mxGetM(mxDirs)==nMaxTerms && mxGetN(mxDirs)==nPreds );  
  CHECK( mxGetClassID(mxCuts)==mxDOUBLE_CLASS );
  CHECK( mxGetM(mxCuts)==nMaxTerms && mxGetN(mxCuts)==nPreds );
  CHECK( mxGetClassID(mxBetas)==mxDOUBLE_CLASS );
  CHECK( mxGetM(mxBetas)==nMaxTerms );  nResp = (int)mxGetN(mxBetas);
  
  // Prepare inputs
  unsigned char *puc = (unsigned char *)mxGetData(mxBestSet);
  bool *BestSet = (bool *)mxMalloc(nMaxTerms*sizeof(bool));  // no native bool type
  for( int i=0; i<nMaxTerms; i++ ) BestSet[i] = puc[i]==0 ? false : true;
  double *pdx = (double *)mxGetData(mxx);
  double *xvec = (double *)mxMalloc(nPreds*sizeof(double));
  
  // Prepare outputs
  mxArray *mxy = mxCreateNumericMatrix(nCases, nResp, mxDOUBLE_CLASS, mxREAL);
  double *pdy = (double *)mxGetData(mxy);
  double *yvec = (double *)mxMalloc(nResp*sizeof(double));
  
  for( int i=0; i<nCases; i++ ) {
    for( int j=0; j<nPreds; j++ ) xvec[j] = pdx[nCases*j+i];  // copy single input vector
    
    // Function call for prediction
    PredictEarth(yvec, xvec, BestSet, (int *)mxGetData(mxDirs), (double *)mxGetData(mxCuts),
                 (double *)mxGetData(mxBetas), nPreds, nResp, nTerms, nMaxTerms);
    
    for( int j=0; j<nResp; j++ ) pdy[nCases*j+i] = yvec[j];  // copy single output vector
  }
  
  // Clean up temporaries
  mxFree(BestSet);
  mxFree(xvec);
  mxFree(yvec);
  
  // Assign and return
  argn = 0;
  CHECK( argn<nlhs );
  plhs[argn++] = mxy;
  
  return;
}

//format(BestSet,Dirs,Cuts,Betas,nTerms,nDigits,MinBeta)
void format(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int argn;
  const mxArray *mxBestSet, *mxDirs, *mxCuts, *mxBetas;
  int nTerms;
  int nMaxTerms, nPreds, nResp;
  int nDigits;
  double MinBeta;
  
  // Input parameters
  argn = 0;
  CHECK( argn<nrhs );  mxBestSet = prhs[argn++];
  CHECK( argn<nrhs );  mxDirs = prhs[argn++];
  CHECK( argn<nrhs );  mxCuts = prhs[argn++];
  CHECK( argn<nrhs );  mxBetas = prhs[argn++];
  CHECK( argn<nrhs );  nTerms = (int)mxGetScalar( prhs[argn++] );
  CHECK( argn<nrhs );  nDigits = (int)mxGetScalar( prhs[argn++] );
  CHECK( argn<nrhs );  MinBeta = (double)mxGetScalar( prhs[argn++] );
  
  // Get input info and do checks
  CHECK( mxGetClassID(mxBestSet)==mxUINT8_CLASS );
  nMaxTerms = (int)mxGetM(mxBestSet);  CHECK( mxGetN(mxBestSet)==1 );
  CHECK( mxGetClassID(mxDirs)==mxINT32_CLASS );
  CHECK( mxGetM(mxDirs)==nMaxTerms );  nPreds = (int)mxGetN(mxDirs); 
  CHECK( mxGetClassID(mxCuts)==mxDOUBLE_CLASS );
  CHECK( mxGetM(mxCuts)==nMaxTerms && mxGetN(mxCuts)==nPreds );
  CHECK( mxGetClassID(mxBetas)==mxDOUBLE_CLASS );
  CHECK( mxGetM(mxBetas)==nMaxTerms );  nResp = (int)mxGetN(mxBetas);
    
  // Prepare inputs
  unsigned char *puc = (unsigned char *)mxGetData(mxBestSet);
  bool *BestSet = (bool *)mxMalloc(nMaxTerms*sizeof(bool));  // no native bool type
  for( int i=0; i<nMaxTerms; i++ ) BestSet[i] = puc[i]==0 ? false : true;
  
  // Function call for output
  FormatEarth(BestSet, (int *)mxGetData(mxDirs), (double *)mxGetData(mxCuts), (double *)mxGetData(mxBetas),
              nPreds, nResp, nTerms, nMaxTerms, nDigits, MinBeta);
  
  // Clean up temporaries
  mxFree(BestSet);
    
  return;
}

// Main mex entry point to despatch functions according to first input value
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int st;
  int argn;
  int mexmethod;
  
  // Input parameters
  argn = 0;
  CHECK( argn<nrhs );
  mexmethod = (int)mxGetScalar( prhs[argn++] );
  
  // Despatch according to value of first argument
  if( mexmethod==0 ) train(nlhs, plhs, nrhs-1, prhs+1);
  else if( mexmethod==1 ) predict(nlhs, plhs, nrhs-1, prhs+1);
  else if( mexmethod==2 ) format(nlhs, plhs, nrhs-1, prhs+1);
  else mexErrMsgTxt("Invalid mexmethod");
  
  return;
}


