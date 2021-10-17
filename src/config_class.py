class ConfigClass(object):
    def __init__(self, configFile):

        ### paths
        paths = configFile['paths']
        self.src_dir = paths['src_dir']
        self.model_dir = paths['model_dir']
        
        ### reinforce        
        reinforce = configFile['reinforce']
        self.episodes = int(reinforce['episodes'])
        self.steps = int(reinforce['steps'])
        
        ### ppo
        ppo = configFile['ppo']
        self.max_ep_len = int(ppo['max_ep_len'])
        self.max_training_timesteps = int(ppo['max_training_timesteps'])
        self.K_epochs = int(ppo['K_epochs'])        
        self.print_freq = self.max_ep_len * 4 # print avg reward in the interval (in num timesteps)
        self.log_freq = self.max_ep_len * 2 # log avg reward in the interval (in num timesteps)  
        self.update_timestep = self.max_ep_len * 4 # update policy every n timesteps
        self.save_model_freq = self.max_ep_len * 4
        self.gamma = float(ppo['gamma'])
        self.clip = float(ppo['clip'])
        self.lr_actor = float(ppo['lr_actor'])
        self.lr_critic = float(ppo['lr_critic'])
        self.nn_number = int(ppo['nn_number'])
        
        ### game
        game = configFile['game']
        self.default_tick_skip = int(game['default_tick_skip'])
        self.physics_ticks_per_second = int(game['physics_ticks_per_second'])
        self.ep_len_seconds = int(game['ep_len_seconds'])
        self.max_steps = int(round(self.ep_len_seconds * self.physics_ticks_per_second / self.default_tick_skip))
        
        ### general
        general = configFile['general']
        self.render = general.getboolean('render')
        self.env = general['env']