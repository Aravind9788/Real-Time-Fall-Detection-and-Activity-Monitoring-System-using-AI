"""
Helper utilities module.

Contains utility functions used across the application.
"""


def rename_person(old_id, new_name, state_manager, all_tracked_people, 
                 manual_id_map, save_manual_id_map_func, data_lock):
    """Safely rename a person and transfer all their stats and mappings.
    
    Args:
        old_id: Old person ID
        new_name: New name for the person
        state_manager: ActivityStateManager instance
        all_tracked_people: Set of all tracked people
        manual_id_map: Dictionary mapping IDs to names
        save_manual_id_map_func: Function to save manual ID mappings
        data_lock: Threading lock for data access
    """
    with data_lock:
        manual_id_map[str(old_id)] = new_name
        
        # Transfer current stats from state_manager
        if state_manager:
            old_stats = state_manager.get_time_stats(old_id)
            new_stats = state_manager.get_time_stats(new_name)
            
            state_manager.walking_time[new_name] = new_stats['walking'] + old_stats['walking']
            state_manager.sitting_time[new_name] = new_stats['sitting'] + old_stats['sitting']
            state_manager.sleeping_time[new_name] = new_stats['sleeping'] + old_stats['sleeping']
            state_manager.standing_time[new_name] = new_stats['standing'] + old_stats['standing']
            
            # Remove old ID stats
            if old_id in state_manager.walking_time:
                del state_manager.walking_time[old_id]
            if old_id in state_manager.sitting_time:
                del state_manager.sitting_time[old_id]
            if old_id in state_manager.sleeping_time:
                del state_manager.sleeping_time[old_id]
            if old_id in state_manager.standing_time:
                del state_manager.standing_time[old_id]
        
        # Update set of all people (remove old ID if it's not the same as new name)
        if str(old_id) in all_tracked_people and str(old_id) != new_name:
            all_tracked_people.remove(old_id)
        all_tracked_people.add(new_name)
        
        # Save mappings immediately
        save_manual_id_map_func()
        print(f"👤 Renamed {old_id} to {new_name} and transferred stats.")
