function waveletCoeffs = cnn_wavelet_decon(images,level,name)

waveletCoeffs  = zeros(size(images,1),size(images,2),sum(2.^level)+1,size(images,3),'double');

for tt= 1:size(images,3)
  coeffs = nsctdec(images(:,:,tt),level,name,'pyr');
  cc = 1;
  waveletCoeffs(:,:,cc,tt) = coeffs{1};
  for l = 1:length(level)
      for ll = 1: 2^(level(l))
          cc = cc+1;
          if level(l) == 0;
              waveletCoeffs(:,:,cc,tt) = coeffs{l+1};
          else
              waveletCoeffs(:,:,cc,tt) = coeffs{l+1}{ll};
          end
      end
  end    
end
end