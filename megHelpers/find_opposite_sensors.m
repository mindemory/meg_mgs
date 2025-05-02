function opposite_sensors = find_opposite_sensors(lay, given_sensors, hemisphere)
    % Extract positions and labels
    pos = lay.pos;
    labels = lay.label;
    midpoint_x = (min(pos(:,1)) + max(pos(:,1))) / 2;
    
    if ~iscell(given_sensors)
        given_sensors = {given_sensors};
    end
    
    opposite_sensors = cell(length(given_sensors),1);
    
    for i = 1:length(given_sensors)
        idx = find(strcmp(labels, given_sensors{i}));
        
        if isempty(idx)
            warning('Sensor %s not found in layout', given_sensors{i});
            continue;
        end
        
        sensor_pos = pos(idx, :);
        
        opposite_pos = sensor_pos;
        opposite_pos(1) = 2 * midpoint_x - sensor_pos(1);
        
        if (hemisphere == "left" && sensor_pos(1) < midpoint_x) || ...
           (hemisphere == "right" && sensor_pos(1) > midpoint_x)
            distances = sum((pos - opposite_pos).^2, 2);
            [~, nearest_idx] = min(distances);
            % opposite_sensors{end+1} = labels{nearest_idx};
            opposite_sensors{i} = labels{nearest_idx};
        else
            warning('Sensor %s is not on the specified hemisphere', given_sensors{i});
        end
    end
end
