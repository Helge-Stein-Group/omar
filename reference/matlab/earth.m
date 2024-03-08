classdef earth
  % Wrapper class for Steve Milborrow's Earth package for multivariate
  % adaptive regression splines.  The interface to the low-level code is
  % via the mex binary earthmex, which must be on the Matlab path and
  % should probably be in the same directory as this file.
  %
  % See the README file in the matlab wrapper directory for building
  % instructions.
  %
  % Fred Nicolls, May 2013

  properties
    nMaxDegree;  % Friedman's mi
    nMaxTerms;  % includes the intercept term
    Penalty;  % GCV penalty per knot
    Thresh;  % forward step threshold
    nMinSpan;  % set to non zero to override internal calculation
    Prune;  % do backward pass
    nFastK;  % Fast MARS K
    FastBeta;  % Fast MARS ageing coef
    NewVarPenalty;  % penalty for adding a new variable
    LinPreds;  % nPreds x 1, 1 if predictor must enter linearly
    UseBetaCache;  % 1 to use the beta cache, for speed
    Trace;  % Display progress: 0 none 1 overview 2 forward 3 pruning 4 more pruning
    
    BestGcv;  % GCV of the best model i.e. BestSet columns of bx
    nTerms;
    BestSet;  % nMaxTerms x 1, indices of best set of cols of bx
    bx;  % nCases x nMaxTerms
    Dirs;  % nMaxTerms x nPreds, -1,0,1,2 for iTerm, iPred
    Cuts;  % nMaxTerms x nPreds, cut for iTerm, iPred
    Residuals;  % nCases x nResp
    Betas;  % nMaxTerms x nPreds, -1,0,1,2 for iTerm, iPred
    
    nDigits;  % format:  number of significant digits to print
    MinBeta;  % format:  terms with fabs(beta) less than this are not printed, 0 for all
  end
  
  methods
    function obj = earth()
      %EARTH  Construct object for regression using Earth (MARS) model
      %  obj = earth() constructs an Earth object for regression using a
      %  MARS model.  Default values are generated for algorithm
      %  parameters, which the user should change if required before calls
      %  to the train, predict, or display methods.
      
      objobj.nMaxDegree = 1;
      obj.nMaxTerms = 21;
      if obj.nMaxDegree>1, obj.Penalty = 3; else obj.Penalty = 2; end
      obj.Thresh = 0.001;
      obj.nMinSpan = 0;
      obj.Prune = 1;
      obj.nFastK = 20;
      obj.FastBeta = 0;
      obj.NewVarPenalty = 0;
      obj.UseBetaCache = 1;
      obj.Trace = 0;
      obj.nDigits = 3;
      obj.MinBeta = 0;
      
      % Outputs from training
      obj.BestGcv = [];  obj.nTerms = [];  obj.BestSet = [];  obj.bx = [];
      obj.Dirs = [];  obj.Cuts = [];  obj.Residuals = [];  obj.Betas = [];
    end
    
    % Train model from samples
    function obj = train(obj,x,y,linpreds)
      %TRAIN  Train earth (MARS) model from samples
      %  obj = train(obj,x,y,linpreds) trains an earth (MARS) model from
      %  inputs x (nCases x nPreds) onto outputs y (nCases x nResp), where
      %  corresponding rows constitute sample pairs.  The optional input
      %  linpreds (nPreds x 1) has values 1 if predictor must enter
      %  linearly (assumed zero otherwise).
      
      error(nargchk(3,4,nargin));
      if nargin==3, linpreds = zeros(size(x,2),1,'int32'); end
      
      % Call mex function for training
      [obj.BestGcv,obj.nTerms,obj.BestSet,obj.bx,obj.Dirs,obj.Cuts, ...
        obj.Residuals,obj.Betas] = ...
        earthmex(0,double(x),double(y),obj.nMaxDegree,obj.nMaxTerms, ...
        obj.Penalty,obj.Thresh, obj.nMinSpan,obj.Prune,obj.nFastK, ...
        obj.FastBeta,obj.NewVarPenalty,int32(linpreds),obj.UseBetaCache, ...
        obj.Trace);
    end
    
    % Predict values for samples
    function y = predict(obj,x)
      %PREDICT  Prediction for earth (MARS) model
      %  y = predict(obj,x) predicts output values y (nCases x nResp) for
      %  earth (MARS) model for input samples x (nCases x nPreds), where
      %  corresponding rows constitute sample pairs.
      
      error(nargchk(2,2,nargin));
      
      % Call mex function for prediction
      y = earthmex(1,x,obj.BestSet,obj.Dirs,obj.Cuts,obj.Betas,obj.nTerms);
    end
    
    % Display current model
    function format(obj)
      %FORMAT  Display formula for earth (MARS) model
      %  format(obj) displays a text representation of the mathematical
      %  formula for the current earth (MARS) model.
      error(nargchk(1,1,nargin));
      
      % Call mex function for display
      earthmex(2,obj.BestSet,obj.Dirs,obj.Cuts,obj.Betas,obj.nTerms,obj.nDigits,obj.MinBeta);
    end
    
  end  % methods
  
end  % classdef