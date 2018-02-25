function [value_list, optimal_position, optimal_value] = pso_optimal(itertimes, variable_number)
    w = 0.8;
    c1 = 2; c2 = 2;
    r1 = 0.25;
    r2 = 0.75;
    particle_number = 1000; % 100������
    %variable_number = variable_number; % 11������
    X = zeros(particle_number, variable_number);
    V = zeros(particle_number, variable_number);
    particle_optimal_position = zeros(particle_number, variable_number);
    optimal_position = zeros(1, variable_number);
    opitmal_value = 1e10;
    x_range = [0,1];
    value_list = [];
    % initial the infomation of each particle
    
    for i = 1:particle_number
        for j = 1:variable_number
            X(i,j) = x_range(1) + (x_range(2) - x_range(1)) * rand();
            V(i,j) = 0;
        end
        X(i,:) = X(i,:) / norm(X(i,:));
        particle_optimal_position(i,:) = X(i,:);
        temp_value = get_Q(X(i,:));
        if(temp_value < opitmal_value)
            optimal_position = X(i,:);
            optimal_value = temp_value;
        end
    end
    count = 0;
    % update the particle
    for iter = 1:itertimes
        for i = 1:particle_number
            V(i,:) = w * V(i,:) + c1 * r1 * (particle_optimal_position(i,:) - ...
                X(i,:)) + c2 * r2 * (optimal_position - X(i,:));
            X(i,:) = X(i,:) + V(i,:);
            X(i,:) = X(i,:) / norm(X(i,:));
            if ( x_range(1) <= min(X(i,:)) && x_range(2) >= max(X(i,:)))
                value_before = get_Q(particle_optimal_position(i,:));
                value_now = get_Q(X(i,:));
                if(value_now < value_before)
                    particle_optimal_position(i,:) = X(i,:);
                end

                if (value_now < optimal_value)
                    optimal_position =X(i,:);
                    optimal_value = value_now;
                    count = count + 1;
                    value_list(count) = optimal_value;
                end
            end
        end
    end
    

end