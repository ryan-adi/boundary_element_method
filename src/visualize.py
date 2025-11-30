from common_modules import np, plt

# plot settings
plt.rcParams.update({
    "figure.figsize": (50,3),   # figure size
    'axes.labelsize': 14,      # Font size for axis labels
    'axes.titlesize': 16,      # Font size for axis titles
    'xtick.labelsize': 12,     # Font size for x-axis ticks
    'ytick.labelsize': 12,     # Font size for y-axis ticks
    #'font.family': 'serif',    # Font family (default is 'sans-serif')
    'font.size': 12,           # General font size for text in the plot
})


# ====================== PRESSURE VISUALIZATION ====================== #
def plot_2d(geoB, p):
    rBid = np.where(np.absolute(geoB['nodes'][:,1]- np.min(geoB['nodes'][:,1]))<10**(-8))[0] # Node indices right side of duct
    evalx,tmpIdx = np.unique(np.round(geoB['nodes'][rBid,0],6),return_index=True)
    idx = rBid[tmpIdx]
    p_abs = np.absolute(p)

    figName = 'Acoustic pressure'
    plt.plot(evalx, p[idx,:].real,'r', evalx,p[idx,:].imag,'b', evalx, p_abs[idx,:],'g')
    plt.xlabel('duct length [m]')
    plt.ylabel('acoustic pressure [Pa]')
    plt.title(figName)
    plt.show()


# ====================== MESH VISUALIZATION ====================== #
def contMeshPlot(contMesh,nodeNumOn,centerPteNumOn,NormOn,view,scalingFactor):
    
    from matplotlib import pyplot

    a_num_nodes = np.asarray(range(contMesh['numNodes']))+1
    a_nodes = contMesh['nodes']
    a_elements = contMesh['elements']
    a_el_centerpoints = contMesh['elcenterpoint']
    a_el_normalvectors = contMesh['elnodalnvec'][0,:,:].transpose()
               
    fig1 = pyplot.figure()
    ax = fig1.add_subplot(111, projection='3d')

    '''shift graph in plot window so that it is somewhat in the middle'''
    fig1.subplots_adjust(left=-0.3, right=1.5, bottom=0.0, top=1.5)

    '''extract X, Y and Z coordinates from node coordinates'''
    X = a_nodes[:,0]
    Y = a_nodes[:,1]
    Z = a_nodes[:,2]

    '''plot now only the nodes'''
    ax.scatter3D(X, Y, Z, c='tab:blue', marker='o', depthshade=False, s=8)
    
    if nodeNumOn:
        for i in range(len(a_num_nodes)):
            '''distinguish between single and double digits'''
            if a_num_nodes[i] < 10:
                label=' %d' %a_num_nodes[i]
            else:
                label = '%d' %a_num_nodes[i]
            '''shift the text a bit so that it fits nicely to the nodes'''
            ax.text(X[i]-0.08, Y[i]-0.01, Z[i]+0.02, label, c='tab:blue', fontdict={'weight': 'ultralight'})

    for k in a_elements:
                
        ax.plot3D(X[k], Y[k], Z[k], c='tab:blue', linewidth=0.8)
        index=np.asarray([k[-1], k[0]])
        ax.plot3D(X[index], Y[index], Z[index], c='tab:blue', linewidth=0.8)
        
    for k in range(len(a_el_centerpoints)):
        
        if centerPteNumOn: 
            if k<9:
                label = ' %d' % (k+1)
            else:
                label = '%d' % (k+1)
                
            if a_el_normalvectors[k,0] == 1:
                ax.text(a_el_centerpoints[k,0], a_el_centerpoints[k,1], a_el_centerpoints[k,2]+0.02,       \
                         label, c='tab:red', fontdict={'weight': 'ultralight'})
            elif a_el_normalvectors[k,1] == -1:
                ax.text(a_el_centerpoints[k,0]-0.02, a_el_centerpoints[k,1]+0.03, a_el_centerpoints[k,2]-0.02,       \
                         label, c='tab:red', fontdict={'weight': 'ultralight'})
    
            elif a_el_normalvectors[k,2] == 1:
                ax.text(a_el_centerpoints[k,0], a_el_centerpoints[k,1], a_el_centerpoints[k,2]-0.02,       \
                         label, c='tab:red', fontdict={'weight': 'ultralight'})
            else:
                ax.text(a_el_centerpoints[k,0]+0.01, a_el_centerpoints[k,1]+0.01, a_el_centerpoints[k,2]-0.01,       \
                        label, c='tab:red', fontdict={'weight': 'ultralight'})
        
        if NormOn: 
            '''plot center nodes and their normal vectors'''
            ax.scatter3D(a_el_centerpoints[k,0], a_el_centerpoints[k,1], a_el_centerpoints[k,2], c='tab:red', marker='o', depthshade=False, s=6)
        
            ax.quiver(a_el_centerpoints[k,0], a_el_centerpoints[k,1], a_el_centerpoints[k,2],     \
                      a_el_normalvectors[k,0], a_el_normalvectors[k,1], a_el_normalvectors[k,2],  \
                      length=scalingFactor, color='tab:red', linewidth=0.5)

    '''
    Set the corresponding limits of the individual axes.
    '''
    ax.set_xlim(-0.03*X.max(), 1.03*X.max())
    ax.set_ylim(-0.25*Y.max(), 1.25*Y.max())
    ax.set_zlim(-0.5*Z.max(), 1.5*Z.max())

    ax.set_xlabel('x', labelpad=6)
    ax.set_ylabel('y', labelpad=6)
    ax.set_zlabel('z')
    
    ax.set_xticks((0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0))
    ax.set_yticks((0, 0.1, 0.2))
    ax.set_zticks((0, 0.1, 0.2))

    ax.tick_params(axis="x", direction="out", length=6)
    ax.tick_params(axis="y", direction="out", length=6)
    ax.tick_params(axis="z", direction="out", length=6)

    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.2, 0.15, 0.5, 1]))

    '''Leave space between axis and graph'''
    ax.use_sticky_edges = False

    '''Remove gray background'''
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    '''Remove gray leftover edge lines'''
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    '''Remove default background grid'''
    ax.grid(False)

    '''
    Adjust the viewing angle onto graph.
    '''
    ax.view_init(35, view)
    
    return ax

def discMeshPlot(contMesh,discMesh,nodeNumOn,discNodeNumOn,NormOn,view,scalingFactor):

    '''
    most parts are similar or same as for the continuous plot which is why
    only the new parts are commented.
    '''

    b_nodes = discMesh['nodes']
    b_nodalnvec = discMesh['nodalnvec']
    font = {'weight': 'ultralight',
            'size': 8}

    ax2 = contMeshPlot(contMesh,nodeNumOn,False,False,view,scalingFactor)
    
    ax2.scatter3D(b_nodes[:,0],b_nodes[:,1], b_nodes[:,2], c='k', marker='o', depthshade=False, s=4)
    
    for k in range(len(b_nodes)):

        if discNodeNumOn:
            if k<9:
                label = '  %d' % (k+1)
            elif 9<=k<=98:
                label = ' %d' % (k+1)
            else:
                label = '%d' % (k+1)

            ax2.text(b_nodes[k,0]+0.01, b_nodes[k,1]+0.01, b_nodes[k,2]-0.01,       \
                         label, c='k', fontdict=font)

        if NormOn:
            ax2.quiver(b_nodes[k,0], b_nodes[k,1], b_nodes[k,2],            \
                       b_nodalnvec[k,0], b_nodalnvec[k,1], b_nodalnvec[k,2],       \
                       length=scalingFactor/2, color='r', linewidth=0.5)