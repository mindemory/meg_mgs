function [left_sensors, right_sensors] = find_mirror_sensors(lay)
    % Extract positions and labels
    pos = lay.pos;
    labels = lay.label;
    midpoint_x = (min(pos(:,1)) + max(pos(:,1))) / 2;
    
    % Initialize arrays for left and right sensors
    left_sensors = {};
    right_sensors = {};
    
    
    % Find pairs of sensors
    for i = 1:length(labels)
        if pos(i,1) > midpoint_x  
            right_sensors{end+1} = labels{i};
        elseif pos(i, 1) < midpoint_x
            left_sensors{end+1} = labels{i};
        end
    end
end
