import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

def make_sample_gif(data, file_path, lng_size = 36, lat_size = 18):
    '''
    data must have:
    1. longitude,
    2. latitude,
    3. DOY
    4. abundance
    '''
    

    
    lng_gird = np.linspace(-180,180,lng_size)
    lat_gird = np.linspace(-90,90,lat_size)[::-1]
        
    fig,ax = plt.subplots()

    def animate(i):
        print(i,end='.')
        ax.clear()
        sub = data[data.DOY==i+1]
        
        sub['lng_grid'] = np.digitize(sub.longitude, lng_gird, right=True)
        sub['lat_grid'] = np.digitize(sub.latitude, lat_gird, right=False)
        sub = sub.groupby(['lng_grid','lat_grid'])[['abundance']].mean().reset_index(drop=False)
        im = np.array([np.nan] * lat_size * lng_size).reshape(lat_size, lng_size)
        im[sub.lat_grid, sub.lng_grid] = sub.abundance
        scat1 = ax.imshow(im, norm=norm)
        
        ax.set_title(f'DOY: {i+1}')
        
        return scat1,
        
    ### scale the color norm
    norm = matplotlib.colors.Normalize(vmin=data.abundance.min(), vmax=data.abundance.max())

    ### for getting the color bar
    scat1 = animate(0)
    fig.colorbar(scat1[0], norm=norm)

    ### animate!
    ani = FuncAnimation(fig, animate, interval=40, blit=True, repeat=True, frames=366)
    ani.save(file_path, dpi=300, writer=PillowWriter(fps=30))
    print()
    print('Finish!')

