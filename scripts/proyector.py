import numpy as np
import cv2


class Proyector:

    def __init__(self, P, R0, V2C):
        R0_padded = np.column_stack([np.vstack([R0, [0, 0, 0]]), [0, 0, 0, 1]])
        V2C_padded = np.vstack((V2C, [0, 0, 0, 1]))
        self.trafo_matrix = np.dot(P, np.dot(R0_padded, V2C_padded))

    def project_and_filter_boxes(self, boxes_3d, img):
        h, w, _ = img.shape
        n = boxes_3d.shape[0]
        projected_boxes = []

        for i in range(n):
            corners_3d = boxes_3d[i]  # (8, 3)
            # Añadir coordenada homogénea
            corners_hom = np.hstack([corners_3d, np.ones((8, 1))])  # (8, 4)
            # Proyección
            pts_2d_homo = (self.trafo_matrix @ corners_hom.T).T  # (8, 3)
            pts_2d = pts_2d_homo[:, :2] / pts_2d_homo[:, 2:3]  # Normalizar por Z

            # Verificar que todos los puntos están dentro de la imagen
            inside = (
                (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) &
                (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
            )
            if np.all(inside):
                projected_boxes.append(pts_2d)

        return np.array(projected_boxes, dtype=np.float32)  # (m, 8, 2)


    def get_2dbboxes(self, proyected_bboxes: np.ndarray) -> np.ndarray:
        """
        args:
            proyected_bboxes (np.ndarray): (m, 8, 2) array with projected 3D bounding boxes
        returns:
            np.ndarray: (m, 4) array with 2D bounding boxes [minx, miny, maxx, maxy]
        """
        if proyected_bboxes.size == 0:
            return np.zeros((0, 4), dtype=np.float32)

        min_xy = proyected_bboxes.min(axis=1)  # (m, 2)
        max_xy = proyected_bboxes.max(axis=1)  # (m, 2)

        # Concatenate into (m, 4): [minx, miny, maxx, maxy]
        bboxes_2d = np.concatenate([min_xy, max_xy], axis=1)  # (m, 4)

        return bboxes_2d.astype(np.float32)

    def proyect(self, points, image):
        proyected_bboxes = self.project_and_filter_boxes(points, image)
        bboxes_2d = self.get_2dbboxes(proyected_bboxes)

        return bboxes_2d
