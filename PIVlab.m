clear all
close all
clc

%Begin with first two frames

%Read image pair 
image_a = im2double(imread('500fps_1000ss_rehaan_maggie_C001H001S0001000001.bmp'));
image_b = im2double(imread('500fps_1000ss_rehaan_maggie_C001H001S0001000002.bmp'));

%Subtract average intensity
image_a = image_a - mean(image_a,'all'); 
image_b = image_b - mean(image_b,'all');

%Select IA size (recommended 64x64 or 48x48)
image_dimensions = size(image_a); %in pixels, assuming square image
image_size = image_dimensions(1);
IA_size = 64; %in pixels
IA_overlap = IA_size*0.25; 

%Setting floor to zero
image_a = image_a - min(image_a,[],'all'); 
image_b = image_b - min(image_b,[],'all');

%Using a Hamming window to eliminate end effects
window = hamming(IA_size)*hamming(IA_size)';

%{
For showing preprocessing difference in report (not applying Hamming 
window)...uncomment to run
%}
% window = IA_size*IA_size;

 
for i = 1:(image_size-IA_size)/IA_overlap+1
    for j = 1:(image_size-IA_size)/IA_overlap+1
       
        window_a = image_a(1+(i-1)*IA_overlap: (i-1)*IA_overlap+IA_size, 1+(j-1)*IA_overlap: (j-1)*IA_overlap+IA_size).*window;
        window_b = image_b(1+(i-1)*IA_overlap: (i-1)*IA_overlap+IA_size, 1+(j-1)*IA_overlap: (j-1)*IA_overlap+IA_size).*window;
        correlation = normxcorr2(window_a, window_b);
        % 
        % %Normalize the correlation
        % ACFTbl1 = autocorr(window_a,10);
        % ACFTbl2 = autocorr(window_b,10);
        % correlation = correlation/(sqrt(ACFTbl1 * ACFTbl2));
        % 
        [x, y] = ndgrid((-IA_size + 1):(IA_size - 1));
        interpolation = griddedInterpolant(x, y, correlation,'spline');
        [peaks, value] = fminsearch(@(f) -interpolation(f(1), f(2)), [0, 0]);

        x_vel(i, j) = peaks(2);
        y_vel(i, j) = peaks(1);
        correlation_value(i, j) = -value; 
        velocity_magnitude(i,j) = sqrt(peaks(2)^2 + peaks(1)^2);

        %Function below found from this forum
        % https://uk.mathworks.com/matlabcentral/answers/277512-how-to-find-peaks-in-3d-mesh
        ix = find(imregionalmax(correlation));
        list = sort(correlation(ix), 'descend');

        correlation_ratios(i,j) = list(1)/list(2);

    end
end


for i = 1:(image_size-IA_size)/IA_overlap+1
    for j = 1:(image_size-IA_size)/IA_overlap+1
        x_pos(i, j) = j;
        y_pos(j, i) = j;
    end
end

figure
quiver(x_pos, y_pos, x_vel, y_vel)
axis ij
title('PIV Relative Velocity Vector Field')
xlabel('x-position')
ylabel('y-position')

figure 
surf(x_pos, y_pos, curl(x_pos, y_pos, x_vel, y_vel))
view(2)
colorbar
title('PIV Relative Vorticity Map')
xlabel('x-position')
ylabel('y-position')
xlim([0,32])
ylim([0,32])

axis ij

figure 
surf(x_pos, y_pos, correlation_value)
view(2)
colorbar
title('Max Correlation Map')
xlabel('x-position')
ylabel('y-position')
axis ij
xlim([0,32])
ylim([0,32])

figure 
surf(x_pos, y_pos, velocity_magnitude)
view(2)
colorbar
title('Velicty Magnitude Plot')
xlabel('x-position')
ylabel('y-position')
axis ij
xlim([0,32])
ylim([0,32])

% %% Run all image pairs
% 
% parameter = "Enter '1' to run all image pairs: ";
% run = input(parameter);
% 
% if run==1
% folder = '/Users/ZoeKing/Desktop/brown university/Grad school /Experimental Fluids';
% f =dir([folder '/*.bmp']);
% n = length(f); %calculate number of files
% pairs = n/2;
% 
% %Loop over all image pairs
% for i =1:pairs
% end
% 
% %Subtract average intensity
% image_a = image_a - mean(image_a(:),'all'); 
% image_b = image_b - mean(image_b(:),'all');
% 
% %Select IA size (recommended 64x64 or 48x48)
% image_size = sqrt(numel(image_a)); %in pixels, assuming square image
% IA_size = 48; %in pixels
% IA_overlap = IA_size*0.25; 
% 
% %Using a Hamming window to eliminate end effects
% window = hamming(IA_size)*hamming(IA_size)';
% 
% for i = 1:(image_size-IA_size)/IA_overlap+1
%     for j = 1:(image_size-IA_size)/IA_overlap+1
% 
%         window_a = image_a(1+(i-1)*IA_overlap: (i-1)*IA_overlap+IA_size, 1+(j-1)*IA_overlap: (j-1)*IA_overlap+IA_size).*window;
%         window_b = image_b(1+(i-1)*IA_overlap: (i-1)*IA_overlap+IA_size, 1+(j-1)*IA_overlap: (j-1)*IA_overlap+IA_size).*window;
%         correlation = normxcorr2(window_a, window_b);
% 
%         [x, y] = ndgrid((-IA_size + 1):(IA_size - 1));
%         interpolation = griddedInterpolant(x, y, correlation,'spline');
%         [peaks, value] = fminsearch(@(f) -interpolation(f(1), f(2)), [0, 0]);
% 
%         x_vel(i, j) = peaks(2);
%         y_vel(i, j) = peaks(1);
%         correlation_value(i, j) = -value; 
%         velocity_magnitude(i,j) = sqrt(peaks(2)^2 + peaks(1)^2);
% 
%         %Function below found from this forum
%         % https://uk.mathworks.com/matlabcentral/answers/277512-how-to-find-peaks-in-3d-mesh
%         ix = find(imregionalmax(correlation));
%         list = sort(correlation(ix), 'descend');
% 
%         correlation_ratios(i,j) = list(1)/list(2);
% 
%     end
% end
% 
% 
% for i = 1:(image_size-IA_size)/IA_overlap+1
%     for j = 1:(image_size-IA_size)/IA_overlap+1
%         x_pos(i, j) = j;
%         y_pos(j, i) = j;
%     end
% end
% 
% figure
% quiver(x_pos, y_pos, x_vel, y_vel)
% axis ij
% title('PIV Relative Velocity Vector Field')
% xlabel('x-position')
% ylabel('y-position')
% 
% figure 
% surf(x_pos, y_pos, curl(x_pos, y_pos, x_vel, y_vel))
% view(2)
% colorbar
% title('PIV Relative Vorticity Map')
% xlabel('x-position')
% ylabel('y-position')
% xlim([0,32])
% ylim([0,32])
% 
% axis ij
% 
% figure 
% surf(x_pos, y_pos, correlation_value)
% view(2)
% colorbar
% title('Max Correlation Map')
% xlabel('x-position')
% ylabel('y-position')
% axis ij
% xlim([0,32])
% ylim([0,32])
% 
% figure 
% surf(x_pos, y_pos, velocity_magnitude)
% view(2)
% colorbar
% title('Velicty Magnitude Plot')
% xlabel('x-position')
% ylabel('y-position')
% axis ij
% xlim([0,32])
% ylim([0,32])
% 
% 
% else 
% end

%% Run all image pairs
parameter = input("Enter '1' to run all image pairs: ", 's');

if strcmp(parameter, '1')
    folder = '/Users/ZoeKing/Desktop/brown university/Grad school /Experimental Fluids';
    f = dir([folder, '/*.bmp']);
    n = length(f); %calculate number of files
    pairs = n / 2;

    % Loop over all image pairs
    for pair_index = 1:pairs
        % Read image pair
        image_a = im2double(imread([folder, '/', f(2*pair_index - 1).name]));
        image_b = im2double(imread([folder, '/', f(2*pair_index).name]));

        % Subtract average intensity
        image_a = image_a - mean(image_a(:));
        image_b = image_b - mean(image_b(:));

        % Select IA size (recommended 64x64 or 48x48)
        image_size = sqrt(numel(image_a)); %in pixels, assuming square image
        IA_size = 48; %in pixels
        IA_overlap = IA_size * 0.25;

        % Initialize variables for this pair
        x_vel = zeros((image_size - IA_size) / IA_overlap + 1);
        y_vel = zeros((image_size - IA_size) / IA_overlap + 1);
        correlation_value = zeros((image_size - IA_size) / IA_overlap + 1);
        velocity_magnitude = zeros((image_size - IA_size) / IA_overlap + 1);
        x_pos = zeros((image_size - IA_size) / IA_overlap + 1);
        y_pos = zeros((image_size - IA_size) / IA_overlap + 1);

        % Processing for each window in the image pair
        % (Add your processing logic here)

        % Plotting or further processing for this pair
        % (Add your plotting logic here)
    end
else
    disp('Invalid input. Exiting.');
end



%% Define threshold for correlation or Q-value
threshold = 0.5; % Adjust as needed

% Initialize variables to keep track of interpolation
interpolated_count = 0;
interpolated_positions = [];

% Loop over each velocity vector
for i = 1:size(x_vel, 1)
    for j = 1:size(x_vel, 2)
        % Check if correlation or Q-value is below threshold
        if correlation_value(i, j) < threshold
            % Perform interpolation
            % Use adjacent vectors in time and space for interpolation
            % Interpolate x_vel(i, j) and y_vel(i, j)
            % Update interpolated_count and interpolated_positions
            
            % Example interpolation (using nearest neighbor)
            x_vel_interp = nearest_neighbor_interpolation(x_vel, i, j);
            y_vel_interp = nearest_neighbor_interpolation(y_vel, i, j);
            
            % Update interpolated count and positions
            interpolated_count = interpolated_count + 1;
            interpolated_positions(interpolated_count, :) = [i, j];
        end
    end
end

% Plot average correlations achieved
average_correlation = mean(correlation_value, 'all');
fprintf('Average correlation value: %.4f\n', average_correlation);

% Plot frequency and distribution of interpolated velocity vectors
figure;
scatter(interpolated_positions(:, 1), interpolated_positions(:, 2), 'filled');
title('Interpolated Velocity Vectors');
xlabel('x-position');
ylabel('y-position');

%% Functions 

function interpolated_value = nearest_neighbor_interpolation(data, i, j)
    % Nearest neighbor interpolation
    i_nearest = round(i);
    j_nearest = round(j);

    % Ensure indices are within bounds
    i_nearest = max(1, min(size(data, 1), i_nearest));
    j_nearest = max(1, min(size(data, 2), j_nearest));

    % Interpolated value
    interpolated_value = data(i_nearest, j_nearest);
end
