package vn.vbi;

import android.content.Context;
import android.graphics.Bitmap;
import android.net.Uri;
import android.provider.MediaStore;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class OpenCvHelper {

    private Context context;

    public OpenCvHelper(Context context) {
        this.context = context;
    }

    public Bitmap getBitmapFromUri(Uri uri) {
        Bitmap bitmap = null;
        try {
            bitmap = MediaStore.Images.Media.getBitmap(context.getContentResolver(), uri);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return bitmap;
    }

    /**
     * Chuyển bitmap sang ma trận opencv
     *
     * @param mat
     * @return
     */
    public Bitmap getBitmapFromMat(Mat mat) {
        int w_proc = mat.width();
        int h_proc = mat.height();
        Bitmap result = Bitmap.createBitmap(w_proc, h_proc, Bitmap.Config.RGB_565);
        Utils.matToBitmap(mat, result);
        return result;
    }

    /**
     * Chuyển ma trận opencv sang bitmap
     *
     * @param bitmap
     * @return
     */
    public Mat getMatFromBitmap(Bitmap bitmap) {
        Mat mat = new Mat();
        Utils.bitmapToMat(bitmap, mat);
        return mat;
    }

    /**
     * Chuyển ma trận thành đen trắng
     *
     * @param origImg
     * @return
     */
    public Mat grayScale(Mat origImg) {
        Mat img = origImg.clone();
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2GRAY);
        return img;
    }

    /**
     * Tìm viền của document trong ảnh
     *
     * @param origImg
     * @return
     */
    public MatOfPoint2f findEdge(Mat origImg) {
        Mat img = origImg.clone();
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2HSV);
        List<Mat> hsv = new ArrayList<>();
        Core.split(img, hsv);
        Mat threshed = new Mat();
        Imgproc.threshold(hsv.get(1), threshed, 50, 255, Imgproc.THRESH_BINARY_INV);
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(threshed, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        MatOfPoint2f contour2f = null;
        MatOfPoint2f biggest = null;
        double max_area = 0;
        for (MatOfPoint contour : contours) {
            contour2f = new MatOfPoint2f(contour.toArray());
            double area = Imgproc.contourArea(contour);
            double peri = Imgproc.arcLength(contour2f, true);
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(contour2f, approx, 0.02 * peri, true);
            if (area > max_area && approx.toArray().length == 4) {
                biggest = approx;
                max_area = area;
            }
        }

        return biggest;
    }

    /**
     * Get hình chữ nhật từ viền đã tìm được
     *
     * @param edge
     * @return
     */
    public Rect findQuadEdge(MatOfPoint2f edge) {
        Point[] points = edge.toArray();
        double xMin = points[0].x;
        double yMin = points[0].y;
        double xMax = xMin;
        double yMax = yMin;
        for (Point point : points) {
            double x = point.x;
            double y = point.y;
            if (x < xMin) xMin = x;
            if (x > xMax) xMax = x;
            if (y < yMin) yMin = y;
            if (y > yMax) yMax = y;
        }
        Rect result = new Rect((int) xMin, (int) yMin, (int) (xMax - xMin), (int) (yMax - yMin));
        return result;
    }

    public void release(Mat... mats) {
        for (Mat mat : mats) {
            if (mat != null) mat.release();
        }
    }
}
