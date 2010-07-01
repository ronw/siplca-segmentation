function X=subsasgn(X,sel,val)
% just a hack to have something to test {} indexing which has no direct
% counterpart in python; we treat it as () indexing, except that the on
% setting we convert from string and on getting to string
switch sel.type,
 case {'{}'}
  sel.type = '()';
  X.data=subsasgn(X.data, sel, str2num(val));
 otherwise
  X.data=subsasgn(X.data, sel, val);
end
end