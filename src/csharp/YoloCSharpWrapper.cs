using System;
using System.Runtime.InteropServices;

namespace Darknet
{
    public class YoloWrapper : IDisposable
    {
        private const string YoloLibraryName = "darknet.dll";
        private const int MaxObjects = 1000;

        public YoloWrapper(string configurationFilename, string weightsFilename, int gpu)
        {
            InitializeYolo(configurationFilename, weightsFilename, gpu);
        }

        public void Dispose()
        {
            DisposeYolo();
        }

        [DllImport(YoloLibraryName, EntryPoint = "init")]
        private static extern int InitializeYolo(string configurationFilename, string weightsFilename, int gpu);

        [DllImport(YoloLibraryName, EntryPoint = "detect_image")]
        private static extern int DetectImage(string filename, ref BboxContainer container);

        [DllImport(YoloLibraryName, EntryPoint = "detect_mat")]
        private static extern int DetectImage(IntPtr pArray, int nSize, ref BboxContainer container);

        [DllImport(YoloLibraryName, EntryPoint = "dispose")]
        private static extern int DisposeYolo();

        public bbox_t[] Detect(string filename)
        {
            var container = new BboxContainer();
            var count = DetectImage(filename, ref container);
            return count == -1 ? null : container.candidates;
        }

        public unsafe bbox_t[] Detect(byte[] imageData)
        {
            var container = new BboxContainer();
            var count = 0;

            fixed (byte* pointer = imageData)
            {
                count = DetectImage((IntPtr)pointer, imageData.Length, ref container);
            }

            return count == -1 ? null : container.candidates;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct bbox_t
        {
            public uint x, y, w, h; // (x,y) - top-left corner, (w, h) - width & height of bounded box
            public float prob; // confidence - probability that the object was found correctly
            public uint obj_id; // class of object - from range [0, classes-1]
            public uint track_id; // tracking id for video (0 - untracked, 1 - inf - tracked object)
            public uint frames_counter;
            public float x_3d, y_3d, z_3d; // 3-D coordinates, if there is used 3D-stereo camera
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct BboxContainer
        {
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = MaxObjects)]
            public bbox_t[] candidates;
        }
    }
}