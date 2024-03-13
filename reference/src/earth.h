void Earth(
    double *pBestGcv,       // out: GCV of the best model i.e. BestSet columns of bx
    int    *pnTerms,        // out: max term nbr in final model, after removing lin dep terms
    bool   BestSet[],       // out: nMaxTerms x 1, indices of best set of cols of bx
    double bx[],            // out: nCases x nMaxTerms
    int    Dirs[],          // out: nMaxTerms x nPreds, -1,0,1,2 for iTerm, iPred
    double Cuts[],          // out: nMaxTerms x nPreds, cut for iTerm, iPred
    double Residuals[],     // out: nCases x nResp
    double Betas[],         // out: nMaxTerms x nResp
    const double x[],       // in: nCases x nPreds
    const double y[],       // in: nCases x nResp
    const double WeightsArg[], // in: nCases x 1, can be NULL, currently ignored
    const int nCases,       // in: number of rows in x and elements in y
    const int nResp,        // in: number of cols in y
    const int nPreds,       // in: number of cols in x
    const int nMaxDegree,   // in: Friedman's mi
    const int nMaxTerms,    // in: includes the intercept term
    const double Penalty,   // in: GCV penalty per knot
    double Thresh,          // in: forward step threshold
    const int nMinSpan,     // in: set to non zero to override internal calculation
    const bool Prune,       // in: do backward pass
    const int nFastK,       // in: Fast MARS K
    const double FastBeta,  // in: Fast MARS ageing coef
    const double NewVarPenalty, // in: penalty for adding a new variable
    const int LinPreds[],       // in: nPreds x 1, 1 if predictor must enter linearly
    const bool UseBetaCache,    // in: 1 to use the beta cache, for speed
    const double Trace,         // in: 0 none 1 overview 2 forward 3 pruning 4 more pruning
    const char *sPredNames[]);  // in: predictor names in trace printfs, can be NULL

void FormatEarth(
    const bool   UsedCols[],// in: nMaxTerms x 1, indices of best set of cols of bx
    const int    Dirs[],    // in: nMaxTerms x nPreds, -1,0,1,2 for iTerm, iPred
    const double Cuts[],    // in: nMaxTerms x nPreds, cut for iTerm, iPred
    const double Betas[],   // in: nMaxTerms x nResp
    const int    nPreds,
    const int    nResp,     // in: number of cols in y
    const int    nTerms,
    const int    nMaxTerms,
    const int    nDigits,   // number of significant digits to print
    const double MinBeta);  // terms with fabs(betas) less than this are not printed, 0 for all

void PredictEarth(
    double       y[],        // out: vector nResp
    const double x[],        // in: vector nPreds x 1 of input values
    const bool   UsedCols[], // in: nMaxTerms x 1, indices of best set of cols of bx
    const int    Dirs[],     // in: nMaxTerms x nPreds, -1,0,1,2 for iTerm, iPred
    const double Cuts[],     // in: nMaxTerms x nPreds, cut for iTerm, iPred
    const double Betas[],    // in: nMaxTerms x nResp
    const int    nPreds,     // in: number of cols in x
    const int    nResp,      // in: number of cols in y
    const int    nTerms,
    const int    nMaxTerms);

