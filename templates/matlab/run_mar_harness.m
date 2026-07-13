function run_mar_harness(dataset_dir, output_dir)
% RUN_MAR_HARNESS  MAR ILS Lab Harness — MATLAB
%   ASTM WKXXXXX Rev 05
%
%   Turnkey harness for applying a lab's MAR algorithm to the ILS sinogram
%   dataset.  The lab implements one function — apply_mar() — in apply_mar.m.
%   This harness handles all HDF5 reading, DICOM writing, directory layout,
%   and CP-2575 MAR macro injection.
%
%   Usage:
%       run_mar_harness('./astm_reference_dataset', './mar_recon')

if nargin < 2, output_dir = './mar_recon'; end

sino_root = fullfile(dataset_dir, 'sinograms');
if ~isfolder(sino_root)
    error('Sinogram directory not found: %s', sino_root);
end

fprintf('MAR ILS Harness\n');
fprintf('  Dataset:  %s\n', dataset_dir);
fprintf('  Output:   %s\n', output_dir);
fprintf('\n');

for cond = {'LP', 'LA'}
    condition = cond{1};
    process_condition(fullfile(sino_root, condition), output_dir, condition);
end

fprintf('\nDone. Submit %s/ to the ILS coordinator.\n', output_dir);
end


function process_condition(sinogram_dir, output_dir, condition)
    h5_files = dir(fullfile(sinogram_dir, 'realization_*.h5'));
    if isempty(h5_files)
        fprintf('  No sinogram files found in %s\n', sinogram_dir);
        return;
    end

    fprintf('  %s: %d realizations\n', condition, length(h5_files));

    for k = 1:length(h5_files)
        h5_path = fullfile(h5_files(k).folder, h5_files(k).name);
        [~, tag, ~] = fileparts(h5_files(k).name);
        fprintf('    %s ... ', tag);

        % Read sinogram and geometry
        [sinogram, geo] = read_sinogram(h5_path);

        % Lab's MAR reconstruction
        hu_volume = apply_mar(sinogram, geo);

        expected = [geo.n_slices, 512, 512];
        if ~isequal(size(hu_volume), expected)
            error('apply_mar returned size [%s], expected [%s]', ...
                num2str(size(hu_volume)), num2str(expected));
        end

        % Write DICOM output
        realization_dir = fullfile(output_dir, condition, tag);
        write_realization(hu_volume, realization_dir, geo.voxel_mm);
        fprintf('done\n');
    end
end


function [sinogram, geo] = read_sinogram(h5_path)
% READ_SINOGRAM  Read sinogram and geometry from an ILS HDF5 file.
%
%   Returns:
%     sinogram — single array, size [n_slices, n_angles, n_det]
%     geo      — struct with fan-beam geometry parameters

    sinogram = single(h5read(h5_path, '/line_integrals'));

    geo.n_slices       = int32(h5readatt(h5_path, '/geometry', 'n_slices'));
    geo.n_angles       = int32(h5readatt(h5_path, '/geometry', 'n_angles'));
    geo.n_det          = int32(h5readatt(h5_path, '/geometry', 'n_det'));
    geo.voxel_mm       = double(h5readatt(h5_path, '/geometry', 'voxel_mm'));
    geo.sid_mm         = double(h5readatt(h5_path, '/geometry', 'SID_mm'));
    geo.sdd_mm         = double(h5readatt(h5_path, '/geometry', 'SDD_mm'));
    geo.gamma_max_deg  = double(h5readatt(h5_path, '/geometry', 'gamma_max_deg'));
    geo.delta_gamma_deg = double(h5readatt(h5_path, '/geometry', 'delta_gamma_deg'));
    geo.angles_deg     = double(h5readatt(h5_path, '/geometry', 'angles_deg'));
    geo.det_fan_angles_deg = double(h5readatt(h5_path, '/geometry', 'det_fan_angles_deg'));
end


function write_realization(hu_volume, output_dir, voxel_mm)
% WRITE_REALIZATION  Write a full HU volume as numbered DICOM CT slices.
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end

    study_uid  = dicomuid();
    series_uid = dicomuid();
    n_slices = size(hu_volume, 1);

    for z = 0:(n_slices - 1)
        write_dicom_slice(squeeze(hu_volume(z+1, :, :)), z, ...
            output_dir, voxel_mm, study_uid, series_uid);
    end
end


function write_dicom_slice(hu_slice, z, output_dir, voxel_mm, study_uid, series_uid)
% WRITE_DICOM_SLICE  Write one 512x512 HU slice as DICOM CT with MAR macro.
    hu_clipped = int16(max(min(hu_slice, 32767), -1024));

    fname = fullfile(output_dir, sprintf('slice_%04d.dcm', z + 1));

    metadata = struct();
    metadata.Modality = 'CT';
    metadata.Manufacturer = 'ASTM WKXXXXX ILS';
    metadata.StudyDescription = 'MAR ILS Lab Submission';
    metadata.SeriesDescription = 'MAR';
    metadata.StudyInstanceUID = study_uid;
    metadata.SeriesInstanceUID = series_uid;
    metadata.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2';
    metadata.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2';
    metadata.Rows = uint16(size(hu_clipped, 1));
    metadata.Columns = uint16(size(hu_clipped, 2));
    metadata.PixelSpacing = [voxel_mm; voxel_mm];
    metadata.ImagePositionPatient = [0; 0; double(z) * voxel_mm];
    metadata.ImageOrientationPatient = [1; 0; 0; 0; 1; 0];
    metadata.SliceLocation = double(z) * voxel_mm;
    metadata.SliceThickness = voxel_mm;
    metadata.InstanceNumber = int32(z + 1);
    metadata.SamplesPerPixel = uint16(1);
    metadata.PhotometricInterpretation = 'MONOCHROME2';
    metadata.BitsAllocated = uint16(16);
    metadata.BitsStored = uint16(16);
    metadata.HighBit = uint16(15);
    metadata.PixelRepresentation = uint16(1);
    metadata.RescaleIntercept = 0;
    metadata.RescaleSlope = 1;

    % DICOM 2026b CP-2575: Metal Artifact Reduction Macro
    % (0018,9390) Metal Artifact Reduction Sequence → (0018,9391) = "YES"
    mar_item = struct();
    mar_item.MetalArtifactReductionApplied = 'YES';
    mar_item.Private_0018_9391 = 'YES';
    metadata.MetalArtifactReductionSequence = mar_item;
    metadata.Private_0018_9390 = mar_item;

    dicomwrite(hu_clipped, fname, metadata, 'CreateMode', 'Copy');
end
