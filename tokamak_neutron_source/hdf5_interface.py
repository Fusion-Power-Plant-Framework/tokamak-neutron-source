from pathlib import Path
from typing import Optional, List
import h5py

import numpy as np
import numpy.typing as npt
import numpy.ma as ma

from tokamak_neutron_source.reactions import AneutronicReactions
from tokamak_neutron_source.reactivity import AllReactions
from tokamak_neutron_source.tools import raw_uc
from tokamak_neutron_source.constants import E_DT_NEUTRON, E_DD_NEUTRON, E_TT_NEUTRON

class Histogram:
    def __init__(self, 
                 name: str,
                 property: str,
                 type: str,
                 value: np.ndarray,
                 dimension: Optional[str] = None,
                 bins: Optional[np.ndarray] = None,
                 interpolation: Optional[str] = None,
                 unit: Optional[str] = None,
                 children: Optional[List[str]] = None,
                 partner: Optional[List[str]] = None):
        # Required
        self.name = name
        self.property = property
        self.type = type
        self.value = value
        # Optional
        self.dimension = dimension
        self.bins = bins
        self.interpolation = interpolation
        self.unit = unit
        self.children = children
        self.partner = partner
    
    def write_hist_to_file(self, file : h5py.File):
        
        group = file.create_group(self.name)
        
        # Attributes
        group.attrs['property'] = self.property
        group.attrs['type'] = self.type
        
        # Attributes (optional)
        if self.dimension is not None:
            group.attrs['dimension'] = self.dimension
        if self.unit is not None:
            group.attrs['unit'] = self.unit
        if self.interpolation is not None:
            group.attrs['interpolation'] = self.interpolation
            
        # Datasets
        group.create_dataset('value', data=self.value)         
        if self.bins is not None:
            group.create_dataset('bins', data=self.bins)
        if self.children is not None:
            group.attrs['parent'] = 'true'
            group.create_dataset('children', data=self.children)
        else:
            group.attrs['parent'] = 'false'
            
        if self.partner is not None:
            group.attrs['married'] = 'true'
            group.create_dataset('partner', data=self.partner)
        else:
            group.attrs['married'] = 'false'
        

def write_hdf5_source(
    file: str | Path,
    r_position: npt.NDArray,
    z_position: npt.NDArray,
    cell_side_length: float,
    temperature: npt.NDArray,
    strength: dict[AllReactions, npt.NDArray],
):
    """
    Write the source (in terms of histograms) to a HDF5 file
    
    Parameters
    ----------
    file:
        The file name stub to which to write the *.h5 file
    r:
        Radial positions of the rings
    z:
        Vertical positions of the rings
    cell_side_length:
        side length of square source cell
    temperature:
        Ion temperatures at the rings [keV]
    strength:
        Dictionary of strengths for each reaction at the rings [arbitrary units]    
    """
    
    # Convert from m to cm
    r_position = raw_uc(r_position, "m", "cm")
    z_position = raw_uc(z_position, "m", "cm")
    half_width = raw_uc(cell_side_length/2, "m", "cm")
    
    # Get list of unique radial positions
    r_bounds = np.unique(r_position)
    
    histograms = {}
    reaction_energy = {'DTn': raw_uc(E_DT_NEUTRON, "J", "MeV"),
                       'DDn': raw_uc(E_DD_NEUTRON, "J", "MeV"),
                       'TTn': raw_uc(E_TT_NEUTRON, "J", "MeV"),}
    
    reaction_name = []
    total_reactivity = [] # This will be the total reactivity for each reaction i.e. bin intensity for decision
    for reaction, react_data in strength.items():
        if reaction not in AneutronicReactions:               
            
            reaction_name.append(reaction.short_name)
            total_reactivity.append(react_data.sum())
            
            # Need to create a scalar to contain the reaction energy and particle type
            hist_name = f'var_{reaction.short_name}.erg_e'
            histograms[hist_name] = Histogram(name=hist_name,
                                              property='energy',
                                              dimension='e',
                                              type='scalar', 
                                              value=np.array([reaction_energy[reaction.short_name]]),
                                              unit='MeV',
                                              children=[f'var_{reaction.short_name}.prt_p'])
            
            hist_name = f'var_{reaction.short_name}.prt_p'
            histograms[hist_name] = Histogram(name=hist_name,
                                              property='particle',
                                              dimension='p',
                                              type='scalar',
                                              value=np.array([1.0]), # neutron
                                              children=[f'var_{reaction.short_name}.pos_r'])
            
            r_values = np.zeros(len(r_bounds))
            for i, r in enumerate(r_bounds):
                # A mask is created because it's applied to multiple arrays (note: 1 is invalid)
                r_mask = [0 if j == r else 1 for j in r_position]
                mx_react = ma.masked_array(react_data, mask=r_mask)
                mx_z = ma.masked_array(z_position, mask=r_mask)
                mx_temp = ma.masked_array(temperature, mask=r_mask)

                # Sum up masked reactivity -> bin value for radius distribution
                r_values[i] = mx_react.sum()
                
                # Create z bin edges from masked position data
                z_bins = [z-half_width for z in mx_z.compressed()]
                z_bins.append(z_bins[-1] + 2*half_width)
                
                # Create z histogram for particular r bin
                hist_name = f'var_{reaction.short_name}.pos_z[{i}]'
                histograms[hist_name] = Histogram(name=hist_name,
                                                    property='position',
                                                    dimension='z',
                                                    type='histogram',
                                                    interpolation='linear',
                                                    bins=np.array(z_bins),
                                                    value=mx_react.compressed(),
                                                    unit='cm',
                                                    partner=[f'erg_temp[{i}]'])
                
                # Create partner temperature histograms
                # This is independent of reaction but needs the radial looping
                hist_name = f'erg_temp[{i}]'
                if hist_name not in histograms.keys():
                    histograms[hist_name] = Histogram(name=hist_name,
                                                        property='energy',
                                                        dimension='temp',
                                                        type='histogram',
                                                        interpolation='linear',
                                                        bins=np.array(z_bins),
                                                        value=mx_temp.compressed(),
                                                        unit='keV')
            
            # Create r bin edges from masked position data
            r_bins = [r-half_width for r in r_bounds]
            r_bins.append(r_bins[-1] + 2*half_width)
            
            hist_name = f'var_{reaction.short_name}.pos_r' 
            histograms[hist_name] = Histogram(name=hist_name,
                                                property='position',
                                                dimension='r',
                                                type='histogram',
                                                interpolation='linear',
                                                bins=np.array(r_bins),
                                                value=r_values,
                                                unit='cm',
                                                children=[f'var_{reaction.short_name}.pos_z[{r}]' for r in range(len(r_values))])


    # Final histogram (and overall grandparent) is the reaction decision
    histograms['var_reaction'] = Histogram(name='var_reaction',
                                       property='variable',
                                       type='decision',
                                       value=np.array(total_reactivity),
                                       bins=np.array(range(len(total_reactivity))),
                                       children= [f'var_{r}.erg_e' for r in reaction_name]
                                       )
    
    
    # Write the histograms to the HDF5 file
    h5_file = h5py.File(f'{file}.h5', 'w')
    for n, h in histograms.items():
        h.write_hist_to_file(h5_file)
    h5_file.close()
    
