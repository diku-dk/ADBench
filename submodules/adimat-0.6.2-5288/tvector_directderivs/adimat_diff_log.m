function varargout = adimat_diff_log(varargin)
  [varargout{1} varargout{2}] = adimat_taylor_logs(varargin{1}, varargin{2}, 'log');
end
% automatically generated from $Id: derivatives-tvdd.xml 4017 2014-04-10 08:55:21Z willkomm $