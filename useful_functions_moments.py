"""
Functions that stay the same between one script and the other.
Used for calculating statistical moments.
"""
import numpy as np

def build_cov_matrix(bb_df,tot_bb):
    """ Calculate the first order moment = centre of mass (com_x and com_y are the coordinates),
    build the covariant matrix from the broken bonds dataframe
     """

    cov_matrix = np.empty([2,2])  # initialise covariant matrix. 2x2
    
    # 1st order moment: position of centre of mass (COM)
    com_x = (bb_df['xcoord100']*bb_df['Broken Bonds']).sum()/tot_bb  # multiplies row by row, then sums everything
    com_y = (bb_df['ycoord100']*bb_df['Broken Bonds']).sum()/tot_bb  # multiplies row by row, then sums everything
    
    #bb_df['x_coord_shifted'] = bb_df['xcoord100']-com_x
    #bb_df['y_coord_shifted'] = bb_df['ycoord100']-com_y
    a = ((bb_df['xcoord100']-com_x)**2*bb_df['Broken Bonds']).sum()/tot_bb  # multiplies tow by row, then sums everything
    b = ((bb_df['xcoord100']-com_x)*(bb_df['ycoord100']-com_y)*bb_df['Broken Bonds']).sum()/tot_bb  # multiplies tow by row, then sums everything
    c = ((bb_df['ycoord100']-com_y)**2*bb_df['Broken Bonds']).sum()/tot_bb  # multiplies tow by row, then sums everything
    
    # a = ((bb_df['x coord']-com_x)**2).sum()/tot_bb  # multiplies tow by row, then sums everything
    # b = ((bb_df['x coord']-com_x)*(bb_df['y coord']-com_y)).sum()/tot_bb  # multiplies tow by row, then sums everything
    # c = ((bb_df['y coord']-com_y)**2).sum()/tot_bb  # multiplies tow by row, then sums everything
    
    cov_matrix[0][0]=a
    cov_matrix[0][1]=b
    cov_matrix[1][0]=b
    cov_matrix[1][1]=c

    return cov_matrix,com_x,com_y


def get_resolution(dir):
    if "200" in dir:
        return "200"
    elif "400" in dir:
        return "400"
    else:
        return "0"