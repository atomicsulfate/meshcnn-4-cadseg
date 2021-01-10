import meshcnn.models.layers.mesh
from tempfile import mkstemp
import os
from shutil import move


class Mesh(meshcnn.models.layers.mesh.Mesh):

    def write_segmented_mesh(self, old_file_path, new_file, segments):
        edge_key = 0
        with open(old_file_path) as old_file:
            for line in old_file:
                if line[0] == 'e':
                    new_file.write('%s %d' % (line.strip(), segments[edge_key]))
                    if edge_key < len(segments):
                        edge_key += 1
                        new_file.write('\n')
                else:
                    new_file.write(line)

    # Extend export segments to export label segments as well.
    def export_segments(self, pred_segments, label_segments):
        if not self.export_folder:
            return
        cur_pred_segments = pred_segments
        cur_label_segments = label_segments
        for i in range(self.pool_count + 1):
            filename, file_extension = os.path.splitext(self.filename)
            file = '%s/%s_%d%s' % (self.export_folder, filename, i, file_extension)
            label_seg_file_path = '%s/%s_%d_labels%s' % (self.export_folder, filename, i, file_extension)
            with open(label_seg_file_path, 'w') as label_seg_file:
                self.write_segmented_mesh(file, label_seg_file, cur_label_segments)
            fh, abs_path = mkstemp()
            with os.fdopen(fh, 'w') as pred_seg_file:
                self.write_segmented_mesh(file, pred_seg_file, cur_pred_segments)
            os.remove(file)
            move(abs_path, file)
            if i < len(self.history_data['edges_mask']):
                cur_pred_segments = pred_segments[:len(self.history_data['edges_mask'][i])]
                cur_pred_segments = cur_pred_segments[self.history_data['edges_mask'][i]]
                cur_label_segments = label_segments[:len(self.history_data['edges_mask'][i])]
                cur_label_segments = cur_label_segments[self.history_data['edges_mask'][i]]