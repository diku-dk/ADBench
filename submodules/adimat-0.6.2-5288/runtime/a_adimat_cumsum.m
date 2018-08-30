% Generated by ADiMat 0.6.0-4867
% © 2001-2008 Andre Vehreschild <vehreschild@sc.rwth-aachen.de>
% © 2009-2013 Johannes Willkomm <johannes.willkomm@sc.tu-darmstadt.de>
% RWTH Aachen University, 52056 Aachen, Germany
% TU Darmstadt, 64289 Darmstadt, Germany
% Visit us on the web at http://www.adimat.de/
% Report bugs to adimat-users@lists.sc.informatik.tu-darmstadt.de
%
%                             DISCLAIMER
% 
% ADiMat was prepared as part of an employment at the Institute for Scientific Computing,
% RWTH Aachen University, Germany and at the Institute for Scientific Computing,
% TU Darmstadt, Germany and is provided AS IS. 
% NEITHER THE AUTHOR(S), THE GOVERNMENT OF THE FEDERAL REPUBLIC OF GERMANY
% NOR ANY AGENCY THEREOF, NOR THE RWTH AACHEN UNIVERSITY, NOT THE TU DARMSTADT,
% INCLUDING ANY OF THEIR EMPLOYEES OR OFFICERS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED,
% OR ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS,
% OR USEFULNESS OF ANY INFORMATION OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE
% WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.
%
% Flags: BACKWARDMODE,  NOOPEROPTIM,
%   NOLOCALCSE,  NOGLOBALCSE,  NOPRESCALARFOLDING,
%   NOPOSTSCALARFOLDING,  NOCONSTFOLDMULT0,  FUNCMODE,
%   NOTMPCLEAR,  DUMP_XML,  PARSE_ONLY,
%   UNBOUND_ERROR
%
% Parameters:
%  - dependents=x
%  - independents=x
%  - inputEncoding=ISO-8859-1
%  - output-mode: plain
%  - output-file: ad_out/a_adimat_cumsum.m
%  - output-file-prefix: 
%  - output-directory: ad_out
% Generated by ADiMat 0.6.0-4867
% © 2001-2008 Andre Vehreschild <vehreschild@sc.rwth-aachen.de>
% © 2009-2013 Johannes Willkomm <johannes.willkomm@sc.tu-darmstadt.de>
% RWTH Aachen University, 52056 Aachen, Germany
% TU Darmstadt, 64289 Darmstadt, Germany
% Visit us on the web at http://www.adimat.de/
% Report bugs to adimat-users@lists.sc.informatik.tu-darmstadt.de
%
%                             DISCLAIMER
% 
% ADiMat was prepared as part of an employment at the Institute for Scientific Computing,
% RWTH Aachen University, Germany and at the Institute for Scientific Computing,
% TU Darmstadt, Germany and is provided AS IS. 
% NEITHER THE AUTHOR(S), THE GOVERNMENT OF THE FEDERAL REPUBLIC OF GERMANY
% NOR ANY AGENCY THEREOF, NOR THE RWTH AACHEN UNIVERSITY, NOT THE TU DARMSTADT,
% INCLUDING ANY OF THEIR EMPLOYEES OR OFFICERS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED,
% OR ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS,
% OR USEFULNESS OF ANY INFORMATION OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE
% WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.
%
% Flags: BACKWARDMODE,  NOOPEROPTIM,
%   NOLOCALCSE,  NOGLOBALCSE,  NOPRESCALARFOLDING,
%   NOPOSTSCALARFOLDING,  NOCONSTFOLDMULT0,  FUNCMODE,
%   NOTMPCLEAR,  DUMP_XML,  PARSE_ONLY,
%   UNBOUND_ERROR
%
% Parameters:
%  - dependents=x
%  - independents=x
%  - inputEncoding=ISO-8859-1
%  - output-mode: plain
%  - output-file: ad_out/a_adimat_cumsum.m
%  - output-file-prefix: 
%  - output-directory: ad_out
%
% Functions in this file: a_adimat_cumsum, rec_adimat_cumsum,
%  ret_adimat_cumsum
%

function [a_x nr_x] = a_adimat_cumsum(x, dim, a_x)
   tmplia1 = 0;
   tmpba1 = 0;
   if nargin<2 || ischar(dim)
      tmpba1 = 1;
      adimat_push1(dim);
      dim = adimat_first_nonsingleton(x);
   end
   adimat_push1(tmpba1);
   len = size(x, dim);
   inds = repmat({':'}, [length(size(x)) 1]);
   inds1 = inds;
   tmpfra1_2 = len;
   for k=2 : tmpfra1_2
      adimat_push_cell_index(inds1, dim);
      inds1{dim} = k - 1;
      adimat_push_cell_index(inds, dim);
      inds{dim} = k;
      adimat_push1(tmplia1);
      tmplia1 = x(inds{:}) + x(inds1{:});
      adimat_push_index(x, inds{:});
      x(inds{:}) = tmplia1;
   end
   adimat_push1(tmpfra1_2);
   nr_x = x;
   a_tmplia1 = a_zeros1(tmplia1);
   if nargin < 3
      a_x = a_zeros1(x);
   end
   tmpfra1_2 = adimat_pop1;
   for k=fliplr(2 : tmpfra1_2)
      x = adimat_pop_index(x, inds{:});
      a_tmplia1 = adimat_adjsum(a_tmplia1, adimat_adjred(tmplia1, adimat_adjreshape(tmplia1, a_x(inds{:}))));
      a_x = a_zeros_index(a_x, x, inds{:});
      tmplia1 = adimat_pop1;
      a_x(inds{:}) = adimat_adjsum(a_x(inds{:}), adimat_adjred(x(inds{:}), a_tmplia1));
      a_x(inds1{:}) = adimat_adjsum(a_x(inds1{:}), adimat_adjred(x(inds1{:}), a_tmplia1));
      a_tmplia1 = a_zeros1(tmplia1);
      inds = adimat_pop_cell_index(inds, dim);
      inds1 = adimat_pop_cell_index(inds1, dim);
   end
   tmpba1 = adimat_pop1;
   if tmpba1 == 1
      dim = adimat_pop1;
   end
end

function x = rec_adimat_cumsum(x, dim)
   tmplia1 = 0;
   tmpba1 = 0;
   if nargin<2 || ischar(dim)
      tmpba1 = 1;
      adimat_push1(dim);
      dim = adimat_first_nonsingleton(x);
   end
   adimat_push1(tmpba1);
   len = size(x, dim);
   inds = repmat({':'}, [length(size(x)) 1]);
   inds1 = inds;
   tmpfra1_2 = len;
   for k=2 : tmpfra1_2
      adimat_push_cell_index(inds1, dim);
      inds1{dim} = k - 1;
      adimat_push_cell_index(inds, dim);
      inds{dim} = k;
      adimat_push1(tmplia1);
      tmplia1 = x(inds{:}) + x(inds1{:});
      adimat_push_index(x, inds{:});
      x(inds{:}) = tmplia1;
   end
   adimat_push(tmpfra1_2, len, inds, inds1, tmplia1, x, x);
   if nargin > 1
      adimat_push1(dim);
   end
   adimat_push1(nargin);
end

function a_x = ret_adimat_cumsum(a_x)
   tmpnargin = adimat_pop1;
   if tmpnargin > 1
      dim = adimat_pop1;
   end
   [x x tmplia1 inds1 inds len] = adimat_pop;
   a_tmplia1 = a_zeros1(tmplia1);
   if nargin < 1
      a_x = a_zeros1(x);
   end
   tmpfra1_2 = adimat_pop1;
   for k=fliplr(2 : tmpfra1_2)
      x = adimat_pop_index(x, inds{:});
      a_tmplia1 = adimat_adjsum(a_tmplia1, adimat_adjred(tmplia1, adimat_adjreshape(tmplia1, a_x(inds{:}))));
      a_x = a_zeros_index(a_x, x, inds{:});
      tmplia1 = adimat_pop1;
      a_x(inds{:}) = adimat_adjsum(a_x(inds{:}), adimat_adjred(x(inds{:}), a_tmplia1));
      a_x(inds1{:}) = adimat_adjsum(a_x(inds1{:}), adimat_adjred(x(inds1{:}), a_tmplia1));
      a_tmplia1 = a_zeros1(tmplia1);
      inds = adimat_pop_cell_index(inds, dim);
      inds1 = adimat_pop_cell_index(inds1, dim);
   end
   tmpba1 = adimat_pop1;
   if tmpba1 == 1
      dim = adimat_pop1;
   end
end
% $Id$