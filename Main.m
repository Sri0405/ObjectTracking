% Read in the video file and extract frames
readerobj = mmreader('video.wmv');
vidFrames = read(readerobj,[500 550]); 

for k = 1 : 50
movl(k).cdata = vidFrames(:,:,:,k);
movl(k).colormap = [];
end

%Initiliaze variables with sizes
thet=zeros(50,1);
trans=zeros(50,1);
scal=zeros(50,1);

%Image preprocessing -base conversion and denoising 
for k=1:50
im=movl(k).cdata;
im1=im;
im2 = rgb2gray(im1);
im3 = im2bw(im2,0.9);
[rows columns] = size(im3);
for i = 1:rows
    for j = 1:columns
        if (im3(i,j)==1)
            imout(i,j) = 0;
        else
            imout(i,j) = 1;
        end
    end
end
 imout1=imfill(imout,'holes');
imout2=imout1-imout;
[rows1 columns1] = size(imout2);
for i = 1:rows1
    for j = 1:columns1
        if (imout2(i,j)==1)
            imout3(i,j) = 0;
        else
            imout3(i,j) = 1;
        end
    end
end
movkl(k).cdata=imout3;
end

%Feature Extraction , Estimate Geometric Transformation using Affine Motion Model ,Ransac Trails for accuracy 
% and threshold check for error due to sunlight or other abnormal parameters

for k=1:49
j=k+1;
imga=movkl(k).cdata;
imgb=movkl(j).cdata;
imga=im2single(imga);
imgb=im2single(imgb);
imgA=movl(k).cdata;
imgB=movl(j).cdata;
imgA=rgb2gray(imgA);
imgB=rgb2gray(imgB);
imgA=im2single(imgA);
imgB=im2single(imgB);
maxPts = 75;
ptThresh = 1e-3;
hCD = vision.CornerDetector( ...
    'Method','Local intensity comparison (Rosen & Drummond)', ...
    'MaximumCornerCount', maxPts, ...
    'CornerThreshold', ptThresh, ...
    'NeighborhoodSize', [9 9]);
pointsA = step(hCD, imga);
pointsB = step(hCD, imgb);
blockSize = 9; 
[featuresA, pointsA] = extractFeatures(imgA, pointsA, ...
    'BlockSize', blockSize);
[featuresB, pointsB] = extractFeatures(imgB, pointsB, ...
    'BlockSize', blockSize);
indexPairs = matchFeatures(featuresA, featuresB, 'Metric', 'SSD');
numMatchedPoints = cast(size(indexPairs, 2), 'int32');
pointsA = pointsA(:, indexPairs(1, :));
pointsB = pointsB(:, indexPairs(2, :));
hGTE = vision.GeometricTransformEstimator(...
                        'Transform','Affine',...
                        'InlierOutputPort',true,...
                        'NumRandomSamplings', 1000);
hGT = vision.GeometricTransformer;
hGTPrj = vision.GeometricTransformer;
nRansacTrials = 1;
Ts = cell(1,nRansacTrials);
costs = zeros(1,nRansacTrials);
nPts = int32(size(pointsA,2));
inliers = cell(1,nRansacTrials);
for j=1:nRansacTrials
    [Ts{j},inliers{j}] = step(hGTE, pointsB, pointsA, nPts);
    imgBp = step(hGT, imgB, Ts{j});
    costs(j) = sum(sum(imabsdiff(imgBp, imgA)));
end
[~,ix] = min(costs);
imgBp = step(hGT, imgB, Ts{ix});
pointsBp = Ts{ix} * [single(pointsB); ones(1,size(pointsB,2))];
H = [Ts{ix}; 0 0 1];
R = H(1:2,1:2);
theta = mean([atan2(R(2),R(1)) atan2(-R(3),R(4))]);
scale = mean(R([1 4])/cos(theta));
translation = H(1:2,3);
HsRt = [scale*[cos(theta) -sin(theta); sin(theta) cos(theta)] translation;
        0 0 1];
 
thet(k,1)=theta;
scal(k,1)=scale;
trans(k,1)=translation(1,1);
 trans(j,1)=translation(2,1);
rootname='new_';
f1=gcf;
filename = [rootname, int2str(k)];
 if (scale<=0.8)
   movg(k).cdata=imgA;
 else
imgBold = step(hGTPrj, imgB, H);
imgBsRt = step(hGTPrj, imgB, HsRt);
movg(k).cdata=imgBsRt;
 end
imshow(movg(k).cdata)
set(f1,'position',[100 100 640 480]); 
print(filename,'-djpeg')
end