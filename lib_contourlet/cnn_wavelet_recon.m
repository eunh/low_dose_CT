function images  = cnn_wavelet_recon(waveletCoeffs,level,name)

waveletCoeffs = double(gather(waveletCoeffs));
images  = (zeros(size(waveletCoeffs,1),size(waveletCoeffs,2),size(waveletCoeffs,4)));
coeffs = cell(1,length(level)+1);
for tt= 1:size(waveletCoeffs,4)  
  cc = 1;
  coeffs{1} = waveletCoeffs(:,:,cc,tt);
  for l = 1:length(level)
      for ll = 1: 2^(level(l))
          cc = cc+1;
          if (level(l)) == 0;
              coeffs{l+1} = waveletCoeffs(:,:,cc,tt);
          else 
              coeffs{l+1}{ll} = waveletCoeffs(:,:,cc,tt);
          end 
      end
  end    
  images(:,:,tt) = nsctrec(coeffs,name,'pyr');
end
end