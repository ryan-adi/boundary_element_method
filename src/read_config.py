from common_modules import ET, np

class Configuration:
    def __init__(self):
        self.config = {}

    def read(self, xml):
        tree = ET.parse(xml)
        root = tree.getroot()

        tags = [child.tag for child in root]
        for tag in tags:
            xml_tag = root.find(tag)
            self.config[tag] = float(xml_tag.text)

        # calculate derived quantities
        self.config['YBC_xMax'] = 1./(self.config['density']*self.config['speedOfSound'])   # [m^3/Pas] acoustic admittance 
        self.config['circularFrequency'] = 2*np.pi*self.config['frequency']                   # [1/s] circular frequency
        self.config['waveLength'] = self.config['speedOfSound']/self.config['frequency']          # [m] wave length
        self.config['waveNumber'] = self.config['circularFrequency']/self.config['speedOfSound']  # [1/m] wave number

    def get(self):
        return self.config