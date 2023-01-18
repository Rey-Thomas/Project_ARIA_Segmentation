from fileinput import filename
import numpy as np
import trimesh
import shapely
from PIL import Image , ImageDraw, ImageChops
import os

# attach to logger so trimesh messages will be printed to console
trimesh.util.attach_to_log()

build_nb='514'
path_maks_matlab='D:/ARIA/Dataset/Image/Photo_recadree_tout_nicolas_matlab/'
path='D:/ARIA/Dataset/Image/'

mesh = trimesh.load('D:/ARIA/Donnee/'+build_nb+'_data/'+build_nb+'_STL.stl')
liste_filename=[]
for f in os.listdir(path+'Photo_recadree/'):
    #if build_nb in f[-37:-33]: #pour tout les builds avant 474
    if build_nb in f[-41:-37]:
      print('True')
      print(f)
      if '_Visible_MeltingEnd' in f:
        print('second TRUE')
        liste_filename.append(f)

print(f'liste filename {liste_filename}')

def rasterize(path,
              pitch,
              origin,
              resolution=None,
              fill=True,
              width=None,
              image_path=None,
              bbx=False,
              mesh=None):
    """
    Rasterize a Path2D object into a boolean image ("mode 1").
    Parameters
    ------------
    path : Path2D
      Original geometry
    pitch : float or (2,) float
      Length(s) in model space of pixel edges
    origin : (2,) float
      Origin position in model space
    resolution : (2,) int
      Resolution in pixel space
    fill :  bool
      If True will return closed regions as filled
    width : int
      If not None will draw outline this wide in pixels
    image_path : str
      Si on veut faire de la transparence entre le mask et l'image réelle
    bbx : bool
      Si on veut dessiner les bounding box des path2D et path3D
    mesh : Path3D
    Returns
    ------------
    raster : PIL.Image
      Rasterized version of input as `mode 1` image
    """

    # check inputs
    pitch = np.asanyarray(pitch, dtype=np.float64)
    origin = np.asanyarray(origin, dtype=np.float64)
    #origin=(-(pitch*resolution[0]/2),-(pitch*resolution[1]/2))


    result = Image.new(mode='1', size=resolution)
    if image_path is not None:
      image = Image.open(image_path)
    draw = ImageDraw.Draw(result)


    # if resolution is None make it larger than path
    if resolution is None:
        span = np.vstack((
            path.bounds, origin)).ptp(axis=0)
        resolution = np.ceil(span / pitch) + 2
    # get resolution as a (2,) int tuple
    resolution = np.asanyarray(resolution,
                               dtype=np.int64)
    resolution = tuple(resolution.tolist())

    bounds = [((i - origin) / pitch).round().astype(np.int64)
                for i in path.bounds]

        
    if mesh is not None:
      mesh_bound_mm=mesh.bounds
      mesh_bounds = [((i[0:2] - origin) / pitch)  .round().astype(np.int64)
                for i in mesh_bound_mm]
      if bbx:
        draw.line([(bounds[0][0],bounds[0][1]),(bounds[0][0],bounds[1][1]),(bounds[1][0],bounds[1][1]),(bounds[1][0],bounds[0][1]),(bounds[0][0],bounds[0][1])] ,fill=1,width=2)
        draw.line([(mesh_bounds[0][0],mesh_bounds[0][1]),(mesh_bounds[0][0],mesh_bounds[1][1]),(mesh_bounds[1][0],mesh_bounds[1][1]),(mesh_bounds[1][0],mesh_bounds[0][1]),(mesh_bounds[0][0],mesh_bounds[0][1])] ,fill=1,width=5)
      centre_path_3d_bbx=[(mesh_bounds[1][0]+mesh_bounds[0][0])/2,(mesh_bounds[1][1]+mesh_bounds[0][1])/2]
        # print(f'bounds_path2d:{bounds}')
        # print(f'bounds 3D:{mesh_bounds}')
        # print(pitch)
        #origin_corrigee=(((bounds[0][0]-mesh_bounds[0][0])*pitch,(bounds[1][1]-mesh_bounds[1][1]))*pitch) 
      origin_corrigee_pixel=(bounds[0][0]-mesh_bounds[0][0],bounds[1][1]-mesh_bounds[1][1]) 
      origin_corrigee_mm=origin_corrigee_pixel*pitch
        #print(f'correction origin pixel:{origin_corrigee_pixel} pour l image: {image_path[-37:-4]}')
        #print(f'correction origin mm:{origin_corrigee} pour l image: {image_path[-37:-4]}')
        #print(f'correction origin mm:{origin_corrigee_mm} pour l image: {image_path[-37:-4]}')
      origin_corrigee=origin+origin_corrigee_mm
        #print(f'origin mm:{origin_corrigee} pour l image: {image_path[-37:-4]}')

        # print(f'origin mm:{origin}')
        #print(f'origin corrige mm:{origin_corrigee}')
        
        # print(f'decalage x:{abs(bounds[0][0]-mesh_bounds[1][0])*pitch}')    
        # print(f'decalage y:{abs(bounds[0][1]-mesh_bounds[1][1])*pitch}')   

        # draw.line([(mesh_bound_mm[0][0]/pitch,centre_path_3d_bbx[1]),(mesh_bound_mm[1][0]/pitch,centre_path_3d_bbx[1])],fill=1)
        # draw.line([(centre_path_3d_bbx[0],mesh_bound_mm[0][1]/pitch),(centre_path_3d_bbx[0],mesh_bound_mm[1][1]/pitch)],fill=1)
        #draw.line([(resolution[0]/2,mesh_bounds[0][1]),(resolution[0]/2,mesh_bounds[1][1])] ,fill=1,width=5)




    # convert all discrete paths to pixel space

    discrete = [((i - origin_corrigee) / pitch).round().astype(np.int64)
                for i in path.discrete]
    # discrete = [((i - origin) / pitch).round().astype(np.int64)
    #             for i in path.discrete]

    # the path indexes that are exteriors
    # needed to know what to fill/empty but expensive
    roots = path.root
    enclosure = path.enclosure_directed
    # draw the exteriors
    
    # if a width is specified draw the outline
    if width is not None:
        width = int(width)
        for coords in discrete:
            draw.line(coords.flatten().tolist(),
                      fill=1,
                      width=width)
        # if we are not filling the polygon exit
        if not fill:
            return result

    # roots are ordered by degree
    # so we draw the outermost one first
    # and then go in as we progress
    for root in roots:
        # draw the exterior
        draw.polygon(discrete[root].flatten().tolist(),
                     fill=1)
        # draw the interior children
        for child in enclosure[root]:
            draw.polygon(discrete[child].flatten().tolist(),
                         fill=0)
    if image_path is not None:
      image.paste(result,(0,0), mask=result)  
      return image
    return result


##UNE TRANCHE
# slice2d= mesh.section([0,0,1],[0,0,-0.9999998])

# slice2d.show()
# slice_2D, _ = slice2d.to_planar()
# slice_2D.show()


# slice2d= mesh.section([0,0,1],[0,0,22.18637])

# slice2d.show()
# slice_2D, _ = slice2d.to_planar()
# slice_2D.show()
print(len(liste_filename))
#liste_filename=[liste_filename[10],liste_filename[20],liste_filename[30],liste_filename[100],liste_filename[200],liste_filename[400]]
z_extents = mesh.bounds[:,2]
z_levels=np.linspace(*z_extents,len(liste_filename))

print(len(z_levels))



# summing the array of Path2D objects will put all of the curves
# into one Path2D object, which we can plot easily
origin = mesh.bounds[0]
print(mesh.bounds)
print(f'origin:{origin[0:2]}')
print(f'origin centroid: {mesh.centroid[0:2]}')
moyenne_origin=[(origin[0]+mesh.centroid[0])/2,(origin[1]+mesh.centroid[1])/2]
print(f'origin moyenne:{moyenne_origin}')
print(mesh.extents)



resolution=(951,951)
taille_plateau=95
pitch=taille_plateau/resolution[0] #200mm de côté pour 951 pixel donc on a la taille d'un pixel en mm, ici le nombre est choisi en gros

facteur=3
pitch=(190*2/resolution[0]/facteur,110*2/resolution[1]/facteur)
pitch=1/10

print(f'pitch (taille en mm d un pixel):{pitch}')

#essai pour décaler l'origine de l'image
origin_corrigee=(origin[0]+1.293+moyenne_origin[0],origin[1]-0.468+moyenne_origin[1]) #valeur de Nicolas pour décalage du laser
origin_corrigee=(origin[0]*1.4+1.293,origin[1]*1.4-0.468) #valeur de Nicolas pour décalage du laser
origin_corrigee=(origin[0]*1.3+1.008-2,origin[1]*1.3-1.016) #valeur de Nicolas pour décalage du laser + des essai pour recaler

# slice_2D.rasterize(pitch=pitch,origin=origin[0:2],resolution=(951,951),fill=True,width=0).save('Mask_sliced_test/test.jpg')#


###Utilise multiplan
# sections = mesh.section_multiplane(plane_origin=mesh.centroid,
#                                    plane_normal=[0,0,1],
#                                    heights=z_levels)
# sections = [mesh.section(plane_origin=mesh.bounds[0],
#                     plane_normal=[0,0,1])
#     	  for h	in z_levels]
# # print(len(sections))
# for i in range(len(sections)):
#     #sections[i].show()
#     #print(mesh.centroid[0:2])
#     #sections[i].show()
#     # print(type(sections[i]))
#     # print(mesh.centroid)
#     # print(np.shape(mesh.centroid))
#     #rasterize permet de créer le mask à partir de chaque tranche réalisée, puis on la sauvgarde
#     #sections[i].rasterize(pitch=(0.15,-0.15),origin=origin[0:2],resolution=(928,928),fill=True,width=0).save(f'Mask_sliced_test/image_test_{i}.jpg')
#     print(type(sections[i]))
#     if isinstance(sections[i],trimesh.path.path.Path2D):
#         sections[i].rasterize(pitch=pitch,origin=origin[0:2],resolution=(951,951),fill=True,width=0).save('Mask_sliced_test/'+liste_filename[i])
#     else:
#         print(sections[i])

#Section_multiplane a des erreur de nonetype(https://github.com/mikedh/trimesh/issues/743), maybe essayer avec juste le truc section
#par section pour eviter les nonetype je sais pas en tout cas il faut essayer

###Utilise le découpage section par section
Obb_done=False


#Matrice homogène pour corriger le défaut de position du laser PB: la translation faite sur le stl n'impacte pas l'image, le centre du stl est tjr (0,0) pour l'image, ainsi faire le décalage sur le stl est inutile
#Reel correction
erreur_decalage_x=0.985
erreur_decalage_y=-1.181


#Correction pour 428 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
if build_nb == '428':
  erreur_decalage_x=1.7
  erreur_decalage_y=-0.5




# Matrice_homogene=np.identity(4)
# Matrice_homogene[:,-1]=[erreur_decalage_x+origin_corrigee[0],erreur_decalage_y+origin_corrigee[1],0,1]
# print(f'Matrice homo:{Matrice_homogene}')
# mesh.apply_transform(Matrice_homogene)
# mesh.show()

# origin_corrigee=(mesh.centroid[0]-(abs(mesh.bounds[0][0])+abs(mesh.bounds[1][0]))/2+erreur_decalage_x,mesh.centroid[1]-(abs(mesh.bounds[0][1])+abs(mesh.bounds[1][1]))/2+erreur_decalage_y)
# origin_corrigee=(mesh.centroid[0]-mesh.extents[0]/2+pitch*resolution[0]/2,mesh.centroid[1]-mesh.extents[1]/2-pitch*resolution[1]/2)
# origin_corrigee=(mesh.extents[0]/2+pitch*resolution[0]/4,-(mesh.extents[1]/2+pitch*resolution[1]/4))
# origin_corrigee=(-mesh.extents[0]/2+pitch*resolution[0]/8,-(-mesh.extents[1]/2+pitch*resolution[1]/8))

# origin_corrigee=(-(mesh.extents[0]/2+pitch*resolution[0]/4+erreur_decalage_x),-(mesh.extents[1]/2+pitch*resolution[1]/4+erreur_decalage_y))
# origin_corrigee=(-(+pitch*resolution[0]/2+erreur_decalage_x),-(pitch*resolution[1]/2-erreur_decalage_y))
# origin_corrigee=(-(+pitch*resolution[0]/2-erreur_decalage_x),-(pitch*resolution[1]/2+erreur_decalage_y))
# print(f'origine corrigée:{origin_corrigee}')

# ### to test if matrice homogène bouge vraiment le stl, faut appuyer sur a pour voir le centre du repère bouger
# Matrice_homogene[:,-1]=[-10,10,0,1]
# mesh.show()
# mesh.apply_transform(Matrice_homogene)
# #slice2d.show()
# mesh.show()

##################################################################################################################################################################
origin_bbx=(-mesh.extents[0]/2,-mesh.extents[1]/2)
print(f'origin bbx: {origin_bbx}')
decalage_x=0
for i in range(len(z_levels)):
    
    z_level=z_levels[i]
    
    #lane_origin=[origin[0],origin[1],z_levels[i]] #marche pas sections return un NoneType
    plane_origin=[0,0,z_level]
    plane_normal=[0,0,1]
    slice2d= mesh.section([0,0,1],[0,0,z_level]) #normal au plan, origine du plan
    
    
    # #on verifie que la fonction section a bien coupé le mesh, car sur le premier ou le dernier le plan est tangeant au mesh et la fonction return un NoneType    
    # if isinstance(slice2d,trimesh.path.path.Path3D): 
    #     slice_2D, _ = slice2d.to_planar()
    #     slice_2D.rasterize(pitch=pitch,origin=origin_corrigee,resolution=(951,951),fill=True,width=0).save('Mask_sliced_test/'+liste_filename[i])
    # else:
    #     print(slice2d)
    
    #on verifie que la fonction section a bien coupé le mesh, car sur le premier ou le dernier le plan est tangeant au mesh et la fonction return un NoneType    
    if isinstance(slice2d,trimesh.path.path.Path3D): 

        #origin_corrigee=(-(pitch*resolution[0]/2+slice2d.extents[0]/16),-(pitch*resolution[1]/2))
        #print(f'origin slice bbx: {-slice2d.extents[0:2]/2}')
        #print(f'origin bbx: {origin_bbx}')
        decalage_centre=origin_bbx-slice2d.extents[0:2]/2
        #print(f'decalage bbx: {decalage_centre}')

        ##ESSAI AVEC LE PATH3D
        centre_correction_x=(slice2d.bounds[0][0]+slice2d.bounds[1][0])/2
        centre_correction_y=(slice2d.bounds[0][1]+slice2d.bounds[1][1])/2
        #print(f'bounds path3D:{slice2d.bounds}')
        #print(f'centre_x: {centre_correction_x}')
        #print(f'centre_y: {centre_correction_y}')
        slice_2D, _ = slice2d.to_planar()

        ##ESSAI AVEC LE PATH2D
        centre_correction_x=(slice_2D.bounds[0][0]+slice_2D.bounds[1][0])
        centre_correction_y=(slice_2D.bounds[0][1]+slice_2D.bounds[1][1])
###SOLUTION MAYBE A REGARDER

        #print(f'bounds path2D:{slice_2D.bounds}')

        # print(f'extents path3D:{slice2d.extents}')
        # print(f'extents path2D:{slice_2D.extents}')
        # print('decalage est fait entre les bounds du path2D et celle du path3D je sais pas pk mais pour moi cest ca qui créer le problème et donc le but va etre d utiliser le bounds du path 3D pour recaler celui du path2D, il y a juste un decalage qui est fait car les dimensions restent les mêmes')
        
        
        
        
        
        ###J'essaye d'utiliser la amtrice OBB, pour recaler le Path2D
        #slice_2D.show()
        # if not Obb_done:
        #     OBB=slice_2D.obb
        #     Obb_done=True
        #     print(OBB)
        #     print('Took a homogenous matrix of reference')
        #     slice_2D.apply_obb()
        #     slice2d= mesh.section([0,0,1],[0,0,z_level])
        #     slice_2D, _ = slice2d.to_planar()
        #     #slice_2D.show()
        # print(f'area:{slice_2D.area}')
        # print(f'medial axis:{(slice_2D.medial_axis)}')
        # print(f'obb:{slice_2D.obb}')
        #slice_2D.apply_obb()
        #img=slice_2D.rasterize(pitch=pitch,origin=origin_corrigee,resolution=(951,951),fill=True,width=0).show()
        # img2=Image.open('D:/ARIA/Dataset/Image/Photo_recadree_tout_nicolas_matlab/'+liste_filename[i])
        #slice_2D.obb=OBB
        #slice_2D.apply_obb()
        #print(slice_2D.root) 
        #print(slice_2D.discrete)
        #print(slice_2D.obb[:,-1]) #les valeur pour replacer le centre du slice correctement par rapport au stl
        # print(slice_2D.obb[0,-1])
        # print(slice_2D.obb[1,-1])
        # print(slice_2D.obb[:,-1])
        #decalage_x+=slice_2D.obb[1,-1]/2
        #origin_corrigee=(-(+pitch*resolution[0]/2-erreur_decalage_x+decalage_x),-(pitch*resolution[1]/2-erreur_decalage_y))



        # origin_corrigee=(-(pitch*resolution[0]/2+erreur_decalage_x-centre_correction_x),-(pitch*resolution[1]/2+erreur_decalage_y-centre_correction_y))
        origin=(-(pitch*resolution[0]/2+erreur_decalage_x),-(pitch*resolution[1]/2+erreur_decalage_y))
        #print(f'origine corrigée:{origin_corrigee}')
        #slice_2D+=slice_2D
        
        
        #Fonctionne
        #img=rasterize(slice_2D,pitch=pitch,origin=origin,resolution=resolution,fill=True,image_path=path+'Photo_recadree/'+liste_filename[i],bbx=True,mesh=slice2d)
        img=rasterize(slice_2D,pitch=pitch,origin=origin,width=None,resolution=resolution,fill=True,image_path=None,bbx=False,mesh=slice2d)

        
        
        
        
        # img=rasterize(slice_2D,pitch=pitch,origin=origin_corrigee,resolution=resolution,fill=True,image_path=path+'Photo_recadree/'+liste_filename[i],bbx=True,mesh=slice2d)

        #img.save('D:/ARIA/Dataset/Mask_parfait/'+liste_filename[i])
        img.save('D:/ARIA/Dataset/Mask_parfait/'+liste_filename[i][:-8]+liste_filename[i][-4:]) #pour 474 et plus
        #slice_2D.rasterize(pitch=pitch,origin=origin_corrigee,resolution=resolution,fill=True,width=0).save('Mask_sliced_test/'+liste_filename[i])

        #   slice_2D.rasterize(pitch= pitch,origin=origin_corrigee,resolution=resolution,fill=True,width=0).save('Mask_sliced_test/'+liste_filename[i][:-4]+'_test.jpg')

    else:
        print(slice2d)


# ### Check si les vertices sont bien placées, pour comprendre pourquoi les images ne se décalle pas:
# oui_3D=True
# slice2d=0
# for i in range(5,505,50):#len(z_levels)):
#     origin_corrigee=(-(+pitch*resolution[0]/2-erreur_decalage_x),-(pitch*resolution[1]/2+erreur_decalage_y))
#     z_level=z_levels[i]
#     plane_origin=[0,0,z_level]
#     plane_normal=[0,0,1]
#     slice2d_a=mesh.section([0,0,1],[0,0,z_level]) #normal au plan, origine du plan
#     if not oui_3D:
#         if isinstance(slice2d_a,trimesh.path.path.Path3D) and i==0: 
#             slice2d =slice2d_a.to_planar()[0]

#         if isinstance(slice2d_a,trimesh.path.path.Path3D) and i!=0: 
#             slice2d_i =slice2d_a.to_planar()[0]
#             slice2d+=slice2d_i
#             #print(slice2d_i.discrete)   

#     if oui_3D:
#         if isinstance(slice2d_a,trimesh.path.path.Path3D) and i==0: 
#             slice2d =slice2d_a
#         if isinstance(slice2d_a,trimesh.path.path.Path3D) and i!=0: 
#             slice2d_i =slice2d_a
#             slice2d+=slice2d_i
#             # print(type(slice2d_i))
#             # print(f'blabla:{slice2d_i.discrete[0]}')  
#             # print(f' type   blabla:{np.shape(slice2d.discrete[0])}')  
#     else:
#         print(slice2d)
    
# slice2d.show()





