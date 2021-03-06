function [times, factors, diffs, types] = admTimingsRev(handle, varargin)

  lastArg = varargin{end};
  if isstruct(lastArg) && isfield(lastArg, 'admopts')
    adopts = lastArg;
    funcArgs = varargin(1:end-1);
  else
    adopts = admOptions;
    funcArgs = varargin;
  end

  adopts.x_types = struct('name', {'fun', ...
                      'Rev/O', 'Rev/D', 'Rev/V' ...
                   }, ...
                          'desc', 'unknown', ...
                          'time', nan, ...
                          'res', [], ...
                          'jac', [], ...
                          'hess', [], ...
                          'isvec', false ...
                          );


  [times, factors, diffs, types] = admGetTimings(handle, funcArgs{:}, adopts);

% $Id: admTimingsRev.m 4251 2014-05-18 20:25:07Z willkomm $
