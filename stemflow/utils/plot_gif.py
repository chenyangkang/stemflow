import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import h3pandas
import geopandas as gpd

def make_sample_gif(data, file_path, col='abundance', 
                            Spatio1='longitude', Spatio2='latitude', Temporal1='DOY',
                            figsize=(18,9), xlims=(-180, 180), ylims=(-90,90), grid=True,
                            lng_size = 360, lat_size = 180, xtick_interval=30, ytick_interval=30,
                            max_frame = 366, log_scale = False, dpi=300, fps=30):
    '''
    data must have:
    1. longitude,
    2. latitude,
    3. DOY
    4. abundance
    
    if log_scale==True, y = np.log(y + 1)
    '''
    
    lng_gird = np.linspace(xlims[0],xlims[1],lng_size)
    lat_gird = np.linspace(ylims[0],ylims[1],lat_size)[::-1]
        
    fig,ax = plt.subplots(figsize=figsize)

    def animate(i, log_scale=log_scale):
        print(i,end='.')
        ax.clear()
        sub = data[data[Temporal1]==i+1]
        
        sub[f'{Spatio1}_grid'] = np.digitize(sub[Spatio1], lng_gird, right=True)
        sub[f'{Spatio2}_grid'] = np.digitize(sub[Spatio2], lat_gird, right=False)
        sub = sub.groupby([f'{Spatio1}_grid', f'{Spatio2}_grid'])[[col]].mean().reset_index(drop=False)

        im = np.array([np.nan] * lat_size * lng_size).reshape(lat_size, lng_size)

        if log_scale:
            im[sub[f'{Spatio2}_grid'].values, sub[f'{Spatio1}_grid'].values] = np.log(sub[col]+1)
        else:
            im[sub[f'{Spatio2}_grid'].values, sub[f'{Spatio1}_grid'].values] = sub[col]
            
        scat1 = ax.imshow(im, norm=norm)
        
        ax.set_title(f'{Temporal1}: {i+1}')
        
        old_x_ticks = np.arange(0,xlims[1] - xlims[0],xtick_interval)
        new_x_ticks = np.arange(xlims[0], xlims[1],xtick_interval)[:len(old_x_ticks)]
        ax.set_xticks(old_x_ticks,
                      new_x_ticks)
        
        old_y_ticks = np.arange(0,ylims[1] - ylims[0],ytick_interval)
        new_y_ticks = np.arange(ylims[0], ylims[1],ytick_interval)[:len(old_y_ticks)]
        ax.set_yticks(old_y_ticks,
                      new_y_ticks)
        
        if grid:
            plt.grid(alpha=0.5)
        plt.tight_layout()
        
        return scat1,
        
    ### scale the color norm
    if log_scale:
        norm = matplotlib.colors.Normalize(vmin=np.log(data[col].min()+1), vmax=np.log(data[col].max()+1))
    else:
        norm = matplotlib.colors.Normalize(vmin=data[col].min(), vmax=data[col].max())

    ### for getting the color bar
    scat1 = animate(0)

    cbar = fig.colorbar(scat1[0], norm=norm, shrink=0.5)
    cbar.ax.get_yaxis().labelpad = 15
    if log_scale:
        cbar.ax.set_ylabel(f'log {col}', rotation=270)
    else:
        cbar.ax.set_ylabel(f'{col}', rotation=270)
    
    if grid:
        plt.grid(alpha=0.5)
    plt.tight_layout()
        
    ### animate!
    ani = FuncAnimation(fig, animate, interval=40, blit=True, repeat=True, frames=max_frame)
    ani.save(file_path, dpi=dpi, writer=PillowWriter(fps=fps))
    print()
    print('Finish!')
    

def make_sample_gif_scatter(data, file_path, col='abundance', 
                            Spatio1='longitude', Spatio2='latitude', Temporal1='DOY',
                            figsize=(18,9), xlims=(-180, 180), ylims=(-90,90), grid=True,
                            max_frame = 366, log_scale = False, s=0.2, dpi=300, fps=30):
    '''
    data must have:
    1. longitude,
    2. latitude,
    3. DOY
    4. abundance
    
    if log_scale==True, y = np.log(y + 1)
    '''
        
    fig,ax = plt.subplots(figsize=figsize)
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])

    def animate(i, log_scale=log_scale):
        print(i,end='.')
        ax.clear()
        sub = data[data[Temporal1]==i+1]
        
        if log_scale:
            sub[col] = np.log(sub[col]+1)
        else:
            pass
            
        scat1 = ax.scatter(sub[Spatio1], sub[Spatio2], s=s, c=sub[col], marker='s')
        
        ax.set_title(f'{Temporal1}: {i+1}')
        ax.set_xlim(xlims[0], xlims[1])
        ax.set_ylim(ylims[0], ylims[1])
        plt.tight_layout()
        if grid:
            plt.grid(alpha=0.5)
        
        return scat1,
        
    ### scale the color norm
    if log_scale:
        norm = matplotlib.colors.Normalize(vmin=np.log(data[col].min()+1), vmax=np.log(data[col].max()+1))
    else:
        norm = matplotlib.colors.Normalize(vmin=data[col].min(), vmax=data[col].max())

    ### for getting the color bar
    scat1 = animate(0)

    cbar = fig.colorbar(scat1[0], norm=norm, shrink=0.5)
    cbar.ax.get_yaxis().labelpad = 15
    if log_scale:
        cbar.ax.set_ylabel(f'log {col}', rotation=270)
    else:
        cbar.ax.set_ylabel(f'{col}', rotation=270)
    plt.tight_layout()
    if grid:
        plt.grid(alpha=0.5)
    
    ### animate!
    ani = FuncAnimation(fig, animate, interval=40, blit=True, repeat=True, frames=max_frame)
    ani.save(file_path, dpi=dpi, writer=PillowWriter(fps=fps))
    print()
    print('Finish!')



def make_sample_gif_hexagon(data, file_path, col='abundance', log_scale = False, H3_RESOLUTION=2, dpi=300, fps=30):
    '''
    data must have:
    1. longitude,
    2. latitude,
    3. DOY
    4. abundance
    
    if log_scale==True, y = np.log(y + 1)
    '''
    if f'h3_0{H3_RESOLUTION}' in data.columns:
        del data[f'h3_0{H3_RESOLUTION}']
        
    u = data[['longitude','latitude']].drop_duplicates()
    u['lng'] = u['longitude']
    u['lat'] = u['latitude']
    u = u.h3.geo_to_h3(H3_RESOLUTION).reset_index(drop=False)
    # u = u[u.geometry.area<200]
    data = data.merge(u, on=['longitude','latitude'], how='left')
    
    uu = u.drop_duplicates(subset=[f'h3_0{H3_RESOLUTION}']).set_index(f'h3_0{H3_RESOLUTION}').h3.h3_to_geo_boundary()
    uu = uu[uu.geometry.area<200]
    # u['val'] = 

    fig,ax = plt.subplots(figsize=(15,15))
    
    def animate(i, log_scale=log_scale, legend=False, return_plot_object=False):
        print(i,end='.')
        ax.clear()
        
        sub = data[data.DOY==i+1]
        sub = sub.groupby([f'h3_0{H3_RESOLUTION}'])[[col]].mean().reset_index(drop=False)
        
        if log_scale:
            sub[col] = np.log(sub[col]+1)
        else:
            pass
        sub = sub.set_index(f'h3_0{H3_RESOLUTION}').h3.h3_to_geo_boundary()
        sub = sub[sub.geometry.area<200]
        scat1 = sub.plot(column=col,
                            # figsize=(15,15),
                            edgecolor='grey',
                            linewidth=0.05,
                            legend=legend,
                            norm=norm,
                            ax=ax,
                            # xlim=(-180, 180),
                            # ylim=(-90, 90)
                        # style_kwds={'shrinkage':0.3},
                            )
        uuu = uu[~uu.index.isin(sub.index)]
        if not len(uuu)==0:
            scat2 = uuu.plot(column=None,
                                # figsize=(15,15),
                                edgecolor=None,
                                linewidth=0,
                                legend=False,
                                alpha=0,
                                ax=ax,
                                )
        
        # Set the extent of the plot
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90,90)
        # ax.axis('off')
        ax.set_title(f'DOY: {i+1}', fontsize=20)
        
        # tt = ax.axes
        if return_plot_object == True:
            return scat1,
        else:
            # ax.axis('off')
            return ax.figure, #scat1.get_children() *scat1, #ax.figure,
        
    ### scale the color norm
    if log_scale:
        norm = matplotlib.colors.Normalize(vmin=np.log(data[col].min()+1), vmax=np.log(data[col].max()+1))
    else:
        norm = matplotlib.colors.Normalize(vmin=data[col].min(), vmax=data[col].max())

    # ### for getting the color bar
    scat1 = animate(0, return_plot_object=True)

    # Add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # Create a divider for existing axes instance
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    sm = scat1[0].get_children()[0]
    cbar = plt.colorbar(sm, cax=cax, norm=norm, shrink=0.3)

    if log_scale:
        cbar.set_label(f'log {col}', rotation=270, labelpad=15)
    else:
        cbar.set_label(f'{col}', rotation=270, labelpad=15)
        
    ### animate!
    from functools import partial
    ani = FuncAnimation(fig, partial(animate, legend=False), interval=40, blit=True, repeat=True, frames=366)
    ani.save(file_path, dpi=dpi, writer=PillowWriter(fps=fps))
    print()
    print('Finish!')

