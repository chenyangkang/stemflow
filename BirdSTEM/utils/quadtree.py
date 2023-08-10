### import libraries
import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
# os.environ['PROJ_LIB'] = r'/usr/proj80/share/proj'
# os.environ['GDAL_DATA'] = r'/beegfs/store4/chenyangkang/miniconda3/share'

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",None)

np.random.seed(42)

from .generate_soft_colors import generate_soft_color


class Point():
    def __init__(self, index, x, y):
        self.x = x
        self.y = y
        self.index = index
        
class Node():
    def __init__(self, x0, y0, w, h, points):
        self.x0 = x0
        self.y0 = y0
        self.width = w
        self.height = h
        self.points = points
        self.children = []

    def get_width(self):
        return self.width
    
    def get_height(self):
        return self.height
    
    def get_points(self):
        return self.points
    
    
def recursive_subdivide(node, grid_len_lon_upper_threshold, grid_len_lon_lower_threshold, \
                            grid_len_lat_upper_threshold, grid_len_lat_lower_threshold, \
                            points_lower_threshold):

    
    if len(node.points)/2 <= points_lower_threshold:
        if not ((node.width > grid_len_lon_upper_threshold) or (node.height > grid_len_lat_upper_threshold)):
            return
    
    if (node.width/2 < grid_len_lon_lower_threshold) or (node.height/2 < grid_len_lat_lower_threshold):
        return
   
    w_ = float(node.width/2)
    h_ = float(node.height/2)

    p = contains(node.x0, node.y0, w_, h_, node.points)
    x1 = Node(node.x0, node.y0, w_, h_, p)
    recursive_subdivide(x1, grid_len_lon_upper_threshold, grid_len_lon_lower_threshold, \
                            grid_len_lat_upper_threshold, grid_len_lat_lower_threshold, \
                            points_lower_threshold)

    p = contains(node.x0, node.y0+h_, w_, h_, node.points)
    x2 = Node(node.x0, node.y0+h_, w_, h_, p)
    recursive_subdivide(x2, grid_len_lon_upper_threshold, grid_len_lon_lower_threshold, \
                            grid_len_lat_upper_threshold, grid_len_lat_lower_threshold, \
                            points_lower_threshold)

    p = contains(node.x0+w_, node.y0, w_, h_, node.points)
    x3 = Node(node.x0 + w_, node.y0, w_, h_, p)
    recursive_subdivide(x3, grid_len_lon_upper_threshold, grid_len_lon_lower_threshold, \
                            grid_len_lat_upper_threshold, grid_len_lat_lower_threshold, \
                            points_lower_threshold)

    p = contains(node.x0+w_, node.y0+h_, w_, h_, node.points)
    x4 = Node(node.x0+w_, node.y0+h_, w_, h_, p)
    recursive_subdivide(x4, grid_len_lon_upper_threshold, grid_len_lon_lower_threshold, \
                            grid_len_lat_upper_threshold, grid_len_lat_lower_threshold, \
                            points_lower_threshold)

    node.children = [x1, x2, x3, x4]
    
    
def contains(x, y, w, h, points):
    pts = []
    for point in points:
        if point.x >= x and point.x <= x+w and point.y>=y and point.y<=y+h:
            pts.append(point)
    return pts


def find_children(node):
    if not node.children:
        return [node]
    else:
        children = []
        for child in node.children:
            children += (find_children(child))
    return children


import matplotlib.pyplot as plt # plotting libraries
import matplotlib.patches as patches

class QTree():
    def __init__(self, grid_len_lon_upper_threshold, grid_len_lon_lower_threshold, \
                        grid_len_lat_upper_threshold, grid_len_lat_lower_threshold, \
                        points_lower_threshold, lon_lat_equal_grid=True,\
                            rotation_angle = 0, \
                    calibration_point_x_jitter = 0,\
                        calibration_point_y_jitter = 0):

        self.points_lower_threshold = points_lower_threshold
        self.grid_len_lon_upper_threshold = grid_len_lon_upper_threshold
        self.grid_len_lon_lower_threshold = grid_len_lon_lower_threshold
        self.grid_len_lat_upper_threshold = grid_len_lat_upper_threshold
        self.grid_len_lat_lower_threshold = grid_len_lat_lower_threshold
        self.lon_lat_equal_grid = lon_lat_equal_grid
        # self.points = [Point(random.uniform(0, 10), random.uniform(0, 10)) for x in range(n)]
        self.points = []
        self.rotation_angle = rotation_angle
        self.calibration_point_x_jitter = calibration_point_x_jitter
        self.calibration_point_y_jitter = calibration_point_y_jitter


    def add_lon_lat_data(self, indexes, x_array, y_array):
        if not len(x_array) == len(y_array) or not len(x_array) == len(indexes):
            raise ValueError("input longitude and latitute and indexes not in same length!")
        
        data = np.array([x_array, y_array]).T
        angle = self.rotation_angle
        r = angle/360
        theta = r * np.pi * 2
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        data = data @ rotation_matrix
        lon_new = (data[:,0] + self.calibration_point_x_jitter).tolist()
        lat_new = (data[:,1] + self.calibration_point_y_jitter).tolist()

        for index,lon,lat in zip(indexes, lon_new, lat_new):
            self.points.append(Point(index, lon, lat))


    
    def generate_griding_params(self):
        x_list = [i.x for i in self.points]
        y_list = [i.y for i in self.points]
        self.grid_length_x = np.max(x_list)-np.min(x_list)
        self.grid_length_y = np.max(y_list)-np.min(y_list)

        left_bottom_point_x = np.min(x_list)
        left_bottom_point_y = np.min(y_list)

        self.left_bottom_point = (left_bottom_point_x ,left_bottom_point_y)
        if self.lon_lat_equal_grid == True:
            self.root = Node(left_bottom_point_x, left_bottom_point_y, \
                max(self.grid_length_x, self.grid_length_y), \
                    max(self.grid_length_x, self.grid_length_y), self.points)
        elif self.lon_lat_equal_grid == False:
            self.root = Node(left_bottom_point_x, left_bottom_point_y, \
                self.grid_length_x, \
                    self.grid_length_y, self.points)
        else:
            raise ValueError('The input lon_lat_equal_grid not a boolean value!')            

    
    def get_points(self):
        return self.points
    
    def subdivide(self):
        recursive_subdivide(self.root, self.grid_len_lon_upper_threshold, self.grid_len_lon_lower_threshold, \
                            self.grid_len_lat_upper_threshold, self.grid_len_lat_lower_threshold, \
                            self.points_lower_threshold)
    
    def graph(self, scatter=True, show=True):
        the_color = generate_soft_color()
        
        plt.figure(figsize=(20, 20))
        plt.xlim([-180,180])
        plt.ylim([-90,90])
        plt.title("Quadtree")
        c = find_children(self.root)
        # print("Number of segments: %d" %len(c))
        areas = set()
        width_set = set()
        height_set = set()
        for el in c:
            areas.add(el.width*el.height)
            width_set.add(el.width)
            height_set.add(el.height)
            
        # print("Minimum segment area: %.3f units^2, min_lon: %.3f units, min_lat: %.3f units" %(min(areas),min(width_set),min(height_set)))
        # print("Maximum segment area: %.3f units^2, max_lon: %.3f units, max_lat: %.3f units" %(max(areas),max(width_set),max(height_set)))

        theta = -(self.rotation_angle/360) * np.pi * 2
        rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
            ])

        for n in c:
            xy0_trans = np.array([[n.x0, n.y0]])
            if self.calibration_point_x_jitter:
                new_x = xy0_trans[:,0] - self.calibration_point_x_jitter
            else:
                new_x = xy0_trans[:,0]
            
            if self.calibration_point_y_jitter:
                new_y = xy0_trans[:,1] - self.calibration_point_y_jitter
            else:
                new_y = xy0_trans[:,1]
            new_xy = np.array([[new_x[0], new_y[0]]]) @ rotation_matrix
            new_x = new_xy[:,0]
            new_y = new_xy[:,1]

            plt.gcf().gca().add_patch(patches.Rectangle((new_x, new_y), n.width, n.height, fill=False,angle=self.rotation_angle, color=the_color))
        
        x = np.array([point.x for point in self.points]) - self.calibration_point_x_jitter
        y = np.array([point.y for point in self.points]) - self.calibration_point_y_jitter

        data = np.array([x,y]).T @ rotation_matrix
        if scatter:
            plt.scatter(data[:,0].tolist(), data[:,1].tolist(), s=0.2, c='tab:blue') # plots the points as red dots
            
        plt.tight_layout()
        plt.gca().set_aspect('equal')
        
        ax = plt.gcf()
        
        ###
        plt.show() if show else plt.close()
        return ax

    def get_final_result(self):
        ## get points assignment to each grid and transform the data into pandas df.
        all_grids = find_children(self.root)
        point_indexes_list = []
        point_grid_width_list = []
        point_grid_height_list = []
        point_grid_points_number_list = []
        calibration_point_list = []
        for grid in all_grids:
            point_indexes_list.append([point.index for point in grid.points])
            point_grid_width_list.append(grid.width)
            point_grid_height_list.append(grid.height)
            point_grid_points_number_list.append(len(grid.points))
            calibration_point_list.append((round(grid.x0, 6), round(grid.y0, 6)))
        
        result = pd.DataFrame({'checklist_indexes': point_indexes_list,
                'stixel_indexes': list(range(len(point_grid_width_list))),
                'stixel_width':point_grid_width_list,
                'stixel_height': point_grid_height_list,
                'stixel_checklist_count':point_grid_points_number_list,
                'stixel_calibration_point(transformed)':calibration_point_list,
                'rotation':[self.rotation_angle] * len(point_grid_width_list),
                'space_jitter(first rotate by zero then add this)':[(round(self.calibration_point_x_jitter, 6), round(self.calibration_point_y_jitter, 6))] * len(point_grid_width_list)})

        result = result[result['stixel_checklist_count']!=0]
        return result
        

def generate_temporal_bins(start, end, step, bin_interval, temporal_bin_start_jitter):
    '''
    start, end, step, bin_interval
    '''
    bin_interval = bin_interval #50
    step = step #20
    
    if type(temporal_bin_start_jitter) == str and temporal_bin_start_jitter=='random':
        jit = np.random.uniform(low=0, high=bin_interval)
    elif type(temporal_bin_start_jitter) in [int, float]:
        jit = temporal_bin_start_jitter
    
    start = start - jit ### ensure 20 DOY
    bin_list = []
    
    i=0
    while True:
        s = start + i * step
        e = s+bin_interval
        if s>=end:
            break
        bin_list.append((s,e))
        i+=1
        
    return bin_list


def get_ensemble_quadtree(data,size=1,
                            grid_len_lon_upper_threshold=25, grid_len_lon_lower_threshold=5,
                            grid_len_lat_upper_threshold=25, grid_len_lat_lower_threshold=5,
                            points_lower_threshold=50,
                            temporal_start = 1, temporal_end=366, temporal_step=20, temporal_bin_interval = 50,
                            temporal_bin_start_jitter = 'random',
                            save_gridding_plot=True,
                            save_path=''):
    '''
    Must have columns:
    1. sampling_event_identifier
    2. DOY
    3. longitude
    4. latitude
    '''
    ensemble_all_df_list = []

    gridding_plot_list = []
        
    for ensemble_count in tqdm(range(size), total=size, desc='Generating Ensemble: '):
        if ensemble_count==0:
            time_jitter = 0
        else:
            time_jitter = -(ensemble_count/size)*30.5
            
        rotation_angle = np.random.uniform(0,360)
        calibration_point_x_jitter = np.random.uniform(-10,10)
        calibration_point_y_jitter = np.random.uniform(-10,10)

        # print(f'ensembel_count: {ensemble_count}')
        
        temporal_bins = generate_temporal_bins(start = temporal_start, 
                                               end=temporal_end, 
                                               step=temporal_step, 
                                                bin_interval = temporal_bin_interval,
                                                temporal_bin_start_jitter = temporal_bin_start_jitter)

        for time_block_index,bin_ in enumerate(temporal_bins):

            time_start = bin_[0]
            time_end = bin_[1]
            sub_data=data[(data['DOY']>=time_start) & (data['DOY']<time_end)]


            QT_obj = QTree(grid_len_lon_upper_threshold=grid_len_lon_upper_threshold, \
                            grid_len_lon_lower_threshold=grid_len_lon_lower_threshold, \
                            grid_len_lat_upper_threshold=grid_len_lat_upper_threshold, \
                            grid_len_lat_lower_threshold=grid_len_lat_lower_threshold, \
                            points_lower_threshold=points_lower_threshold, \

                            lon_lat_equal_grid = True, rotation_angle = rotation_angle, \
                                calibration_point_x_jitter = calibration_point_x_jitter,\
                                    calibration_point_y_jitter = calibration_point_y_jitter)

            ## Give the data and indexes. The indexes should be used to assign points data so that base model can run on those points,
            ## You need to generate the splitting parameters once giving the data. Like the calibration point and min,max.
            
            # print(sub_data.index, sub_data['longitude'].values, sub_data['latitude'].values)
            QT_obj.add_lon_lat_data(sub_data.index, sub_data['longitude'].values, sub_data['latitude'].values)
            QT_obj.generate_griding_params()
            
            ## Call subdivide to precess
            QT_obj.subdivide()
            this_slice = QT_obj.get_final_result()
            
            if save_gridding_plot:
                ax = QT_obj.graph(scatter=False, show=False)
                gridding_plot_list.append({
                    'ensemble':ensemble_count,
                    'time_block_index':time_block_index,
                    'time_bin_start':bin_[0],
                    'time_bin_end':bin_[1],
                    'ax':ax
                })
                
            this_slice['ensemble_index'] = ensemble_count
            this_slice['DOY_start'] = time_start
            this_slice['DOY_end'] = time_end
            # this_slice['checklist_name'] = [sub_data.loc[i,:]['sampling_event_identifier'].values.tolist() for i in this_slice['checklist_indexes']]
            this_slice['DOY_start']=round(this_slice['DOY_start'],1)
            this_slice['DOY_end']=round(this_slice['DOY_end'],1)
            this_slice['unique_stixel_id'] = [str(time_block_index)+"_"+str(i)+"_"+str(k) for i,k in zip (this_slice['ensemble_index'].values, 
                                                                                                          this_slice['stixel_indexes'].values)]
            ensemble_all_df_list.append(this_slice)
            
    ensemble_df = pd.concat(ensemble_all_df_list).reset_index(drop=True)
    if not save_path=='':
        ensemble_df.to_csv(save_path,index=False)
        print(f'Saved! {save_path}')
        
    return ensemble_df, gridding_plot_list

