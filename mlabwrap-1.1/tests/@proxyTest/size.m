function varargout=size(X,varargin)
[varargout{1:nargout}] = size(X.data,varargin{:});
end