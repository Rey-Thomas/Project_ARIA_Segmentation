%Déclaration paramètres
machine_ref_points=[[190;-110;1],[190;110;1],[-190;110;1],[-190;-110;1]]; %Coordonnées dans Rt en mm des points de référence
pictures_ref_points=[[1936.22;1726.25;1],[1927.99;664.57;1],[27.46;546.06;1],[-2.57;1745.55;1]]; %Coordonnées en pixel des points de référence sur les photos visibles
scale_factor=10; %Facteur d'échelle pour le redressement des photos lors des reconstructions (pixels/mm)
histeq_activity=true;
threshold_sensitivity=0.45; %Facteur de sensibilté pour la binarisation de l'image
hsize=8; %Voir fspecial pour gaussian (pixels)
sigma=8; %Voir fspecial pour gaussian (pixels)
threshold_multiplier=1; %Multiplié à threshold, voir threshold dans edge pour sobel
len=1; %Voir strel pour line (pixels)
gapsize=1; %Voir filledgegaps (pixels)
diamond_size=1; %Taille du diamant pour l'erodage de l'image
erode_number=3; %Nombre d'erodage de l'image
P=150; %Voir bwareaopen (pixels)


%IL FAUT FAIRE QUE LE WIDTH ET LE HEIGHT DES IMAGES SOIT DIVISIBLE PAR 32 SOIT 960 SOIT 928 A CAUSE DES SKIP CONNECTIONS    


%Importation données
pathname=[uigetdir '\'];
filenames={dir(pathname).name};
filenames(strcmp(filenames,'.') | strcmp(filenames,'..'))=[];

nb_build=inputdlg('Numéro du build');


%Calcul paramètres homographie
pic=[];
for u=1:length(filenames)%_computed)
        %pic=imread([pathname filenames_computed{u}]); %Read a first image to compute data for point translation after homography
        pic=imread([pathname filenames{u}]); %Read a first image to compute data for point translation after homography
        size_ref=size(pic);
        break;
end

    %Determination coordonnées natives du centre tête théorique
    x1=[pictures_ref_points(1,1) pictures_ref_points(1,3)];
    y1=[pictures_ref_points(2,1) pictures_ref_points(2,3)];
    x2=[pictures_ref_points(1,2) pictures_ref_points(1,4)];
    y2=[pictures_ref_points(2,2) pictures_ref_points(2,4)];
    p1=polyfit(x1,y1,1);
    p2=polyfit(x2,y2,1);
    x_intersect=fzero(@(x) polyval(p1-p2,x),3);
    y_intersect=polyval(p1,x_intersect);
    native_center_plate_coordinates=[x_intersect;y_intersect];
    
    %Détermination de la transformation géométrique
    P1=[pictures_ref_points(:,1),pictures_ref_points(:,2),pictures_ref_points(:,3),pictures_ref_points(:,4)];
    P2=[scale_factor*[machine_ref_points(1:2,1),machine_ref_points(1:2,2),machine_ref_points(1:2,3),machine_ref_points(1:2,4)];...
        [machine_ref_points(3,1),machine_ref_points(3,2),machine_ref_points(3,3),machine_ref_points(3,4)]];
    H=homography2d(P1,P2);
    
    tform=projective2d(transpose(H));
    tformbis=tform;
    
    
    %Transformation par homographie
    transformed_pictures_ref_points=H*P1;
    transformed_center_plate_coordinates=H*[native_center_plate_coordinates;1];
    
    %Recalage
    Rin=imref2d(size_ref);
    P_origine=H*[[min(Rin.XWorldLimits);min(Rin.YWorldLimits);1],[min(Rin.XWorldLimits);max(Rin.YWorldLimits);1],[max(Rin.XWorldLimits);min(Rin.YWorldLimits);1],[max(Rin.XWorldLimits);max(Rin.YWorldLimits);1]];
    P_origine=P_origine./repmat(P_origine(3,:),[3,1]);
    trans=[min(P_origine(1,:));min(P_origine(2,:))];
    
    corrected_ref_points=round(transformed_pictures_ref_points(1:2,:)./transformed_pictures_ref_points(3,:)-repmat(trans,1,size(transformed_pictures_ref_points,2)),3);
    corrected_center_plate_coordinates=round(transformed_center_plate_coordinates(1:2)./transformed_center_plate_coordinates(3)-trans,3);

    %[native_center_plate_coordinates,tform,corrected_center_plate_coordinates,corrected_ref_points]=deal([],[],[],[]);
    %disp('Impossible de calculer les paramètres de redressement des photos avec les réglages courants');
    %return


corrected_picture=imwarp(pic,tform,'cubic'); %Homographie
%imshow(corrected_picture)

%Définition ROI
%answer=inputdlg(['Coordonnées [Xmin,Xmax,Ymin,Ymax] (pixels) :' ],'Définition manuelle de la zone d''intérêt',[1 70],{'[1497,2447,2090,3040]'});
answer=[1497,2447,2090,3040];%str2num(answer{:});
col=[answer(1);answer(2)];
row=[answer(3);answer(4)];

%Préparation calcul
G=fspecial('Gaussian',hsize,sigma); %Calculate predifined 2D filter using Gaussian kernel with a blur width (standard deviation)
dil_param=[strel('line',len,90) strel('line',len,0)];
diamond_size=strel('diamond',diamond_size); %Création de l'outil diamant

%Calcul
for j=1:length(filenames)
        img=imread([pathname filenames{j}]);
        if isequal(size_ref,size(img))
            img=imwarp(img,tform,'cubic'); %Homography
            img=img(row(1):row(2),col(1):col(2),:); %Cutting according to ROI

            filename_bis=filenames(j);
            name_img=strcat('Dataset/Image/Photo_recadree/',strcat(strcat(nb_build{1},'_'),filename_bis{1}));
            imwrite(img,name_img); % ENREGSITREMENT IMAGE


%             if size(img,3)==3
%                 img=rgb2gray(img); %Convert RGB image into grayscale image
%             end
%             if histeq_activity
%                 img=histeq(img); %Lissage de l'histogramme du contraste
%             end
%             img=imbinarize(img,'adaptive','ForegroundPolarity','dark','Sensitivity',threshold_sensitivity); %Convert grayscale image into binary image using a locally adaptive threshold value
%             img=imfilter(img,G,'replicate'); %Filtering N-D multidimensions image with Gaussian option
%             [~,threshold]=edge(img,'sobel','thinning'); %Compute the threshold value for edge detection process
%             img=edge(img,'sobel',threshold*threshold_multiplier); %Egde detection ('sobel' option) with the threshold value
%             img=imdilate(img,dil_param); %Dilate image
%             img=filledgegaps(img,gapsize); %Fill small edge gaps in the detected binary image, change and get the smallest value. Should choose its value is odd
%             img=imfill(img,'holes'); %Fill the regions and holes in the image
%             
%             if erode_number~=0
%                 for k=1:erode_number
%                     img=imerode(img,diamond_size); %Érodage de l'image
%                 end
%             end
%             img=bwareaopen(img,P); %Remove connected areas which have less than TBR_connected_area_size pixels /!\ Important /!\
%             name_img=strcat('Dataset/Image/Photo_recadree_tout_nicolas_matlab/',strcat(strcat(nb_build{1},'_bis'),filename_bis{1}));
%             imwrite(img,name_img); % ENREGSITREMENT IMAGE
%             %data(:,:,layer_numbers(j))=img; %V dans isosurface
          end
end
