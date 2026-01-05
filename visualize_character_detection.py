"""
Sanskrit OCR Visualization Script
This script generates comprehensive visualizations including:
1. Character-level heatmaps using CRAFT model
2. Bounding boxes around each character
3. Line segmentation visualization
"""

import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from skimage import io
import torch.nn as nn
import torch.nn.init as init
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import namedtuple
from packaging import version
from collections import OrderedDict
from scipy.signal import find_peaks

# ============================================================================
# CRAFT MODEL DEFINITION
# ============================================================================

def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

class vgg16_bn(torch.nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(vgg16_bn, self).__init__()
        if version.parse(torchvision.__version__) >= version.parse('0.13'):
            vgg_pretrained_features = models.vgg16_bn(
                weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None
            ).features
        else:
            models.vgg.model_urls['vgg16_bn'] = models.vgg.model_urls['vgg16_bn'].replace('https://', 'http://')
            vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(12):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
            
        self.slice5 = torch.nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                nn.Conv2d(1024, 1024, kernel_size=1)
        )

        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())

        init_weights(self.slice5.modules())

        if freeze:
            for param in self.slice1.parameters():
                param.requires_grad= False

    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out

class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class CRAFT(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()
        self.basenet = vgg16_bn(pretrained, freeze)
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        sources = self.basenet(x)
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)
        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)
        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)
        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)
        y = self.conv_cls(feature)
        return y.permute(0,2,3,1), feature

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def loadImage(img_file):
    img = io.imread(img_file)
    if img.shape[0] == 2: 
        img = img[0]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:,:,:3]
    img = np.array(img)
    return img

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    img = in_img.copy().astype(np.float32)
    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def detect(img, detector, device):
    x = [np.transpose(normalizeMeanVariance(img), (2, 0, 1))]
    x = torch.from_numpy(np.array(x))
    x = x.to(device)
    with torch.no_grad():
        y, feature = detector(x)
    d = y[0,:,:,0].cpu().data.numpy()
    return d

def gen_bounding_boxes(det, peaks):
    img = np.uint8(det * 255)
    _, img1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    max_height = np.percentile(peaks[1:]-peaks[:-1], 80)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h <= max_height:
            bounding_boxes.append((x, y, w, h))
        else:
            n_b = np.int32(np.ceil(h/max_height))
            equal_height = h // n_b
            height_adjustment = h - (equal_height * n_b)
            for i in range(n_b):
                new_y = y + (i * equal_height)
                box_height = equal_height + (height_adjustment if i == n_b - 1 else 0)
                bounding_boxes.append((x, new_y, w, box_height))
    
    return bounding_boxes

def assign_lines(bounding_boxes, det):
    ys = det.sum(axis=1)
    thres = 0.5 * ys.max()
    peaks, _ = find_peaks(ys, height=thres, distance=det.shape[0]/100, width=5)
    
    xs = det.sum(axis=0)
    thres = 0.5 * xs.max()
    xpeaks, _ = find_peaks(xs, height=thres)
    
    # Find peaks at different x positions
    ys1 = det[:,xpeaks[0]:xpeaks[0]+100].sum(axis=1)
    thres = 0.5 * ys1.max()
    p1, _ = find_peaks(ys1, height=thres, distance=det.shape[0]/100, width=5)
    
    ys2 = det[:,xpeaks[-1]-100:xpeaks[-1]].sum(axis=1)
    thres = 0.5 * ys2.max()
    p2, _ = find_peaks(ys2, height=thres, distance=det.shape[0]/100, width=5)
    
    xmid = int((xpeaks[0]+xpeaks[-1])/2)
    ys3 = det[:,xmid-50:xmid+50].sum(axis=1)
    thres = 0.5 * ys3.max()
    p3, _ = find_peaks(ys3, height=thres, distance=det.shape[0]/100, width=5)
    
    if(peaks[0]-p1[0]>det.shape[0]/12):
        p1 = np.copy(p1[1:])
    
    p = min(p1, p2, p3, key=len)
    l = len(p)
    
    if(len(p1)>=l+1):
        k = len(p1) - len(p)
        ind = np.argmin(np.abs(p1[:k+1] - p[0]))
        peaks1 = p1[ind:l+ind]
    else:
        peaks1 = p1

    if(len(p2)>=l+1):
        k = len(p2) - len(p)
        ind = np.argmin(np.abs(p2[:k+1] - p[0]))
        peaks2 = p2[ind:l+ind]
    else:
        peaks2 = p2

    if(len(p3)>=l+1):
        k = len(p3) - len(p)
        ind = np.argmin(np.abs(p3[:k+1] - p[0]))
        peaks3 = p3[ind:l+ind]
    else:
        peaks3 = p3

    lines = []
    for box in bounding_boxes:
        x, y, _, h = box
        mid_y = y + h / 2
        wt1 = np.abs(x - xpeaks[0])
        wt2 = np.abs(x - xpeaks[-1])
        wt3 = np.abs(x - xmid)
        if x<=xmid:
            peaks_interp = wt3*peaks1/(wt1+wt3)+wt1*peaks3/(wt1+wt3)
        else:
            peaks_interp = wt3*peaks2/(wt2+wt3)+wt2*peaks3/(wt2+wt3)
        
        c_index = np.argmin(np.abs(peaks_interp - mid_y))
        if(np.abs(mid_y-peaks_interp[c_index])>20):
            c_index=-1
        lines.append(c_index)
    
    return lines, peaks1

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_all(image_path, output_dir='visualization_output'):
    """
    Complete visualization pipeline for Sanskrit OCR
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CRAFT model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    _detector = CRAFT()
    model_path = './line-segmentation/pretrained_craft/craft_mlt_25k.pth'
    _detector.load_state_dict(copyStateDict(torch.load(model_path, map_location=device)))
    detector = torch.nn.DataParallel(_detector).to(device)
    detector.eval()
    
    # Load image
    print(f"Loading image: {image_path}")
    original_image = loadImage(image_path)
    
    # Generate heatmap
    print("Generating character heatmap...")
    heatmap = detect(original_image, detector, device)
    
    # Find peaks for line detection
    ys = heatmap.sum(axis=1)
    thres = 0.5 * ys.max()
    peaks, _ = find_peaks(ys, height=thres, distance=heatmap.shape[0]/100, width=5)
    
    # Generate bounding boxes
    print("Generating character bounding boxes...")
    bounding_boxes = gen_bounding_boxes(heatmap, peaks)
    
    # Assign lines
    print("Assigning characters to lines...")
    lines, line_peaks = assign_lines(bounding_boxes, heatmap)
    
    # ============================================================================
    # VISUALIZATION 1: Original Image with Bounding Boxes
    # ============================================================================
    print("Creating visualization 1: Bounding boxes on original image...")
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Resize original for display
    display_img = cv2.resize(original_image, heatmap.shape[::-1])
    display_img_gray = cv2.cvtColor(display_img, cv2.COLOR_RGB2GRAY)
    
    ax.imshow(display_img_gray, cmap='gray')
    
    # Draw bounding boxes with different colors for each line
    colors = plt.cm.rainbow(np.linspace(0, 1, len(line_peaks)))
    for (x, y, w, h), line_idx in zip(bounding_boxes, lines):
        if line_idx >= 0:
            color = colors[line_idx]
            rect = Rectangle((x, y), w, h, linewidth=1.5, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
    
    ax.set_title('Character Bounding Boxes (Each line in different color)', fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/1_bounding_boxes.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/1_bounding_boxes.png")
    plt.close()
    
    # ============================================================================
    # VISUALIZATION 2: Character Heatmap
    # ============================================================================
    print("Creating visualization 2: Character heatmap...")
    fig, ax = plt.subplots(figsize=(20, 10))
    
    im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
    ax.set_title('Character Region Heatmap (CRAFT Model Output)', fontsize=16)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/2_character_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/2_character_heatmap.png")
    plt.close()
    
    # ============================================================================
    # VISUALIZATION 3: Heatmap Overlay on Original
    # ============================================================================
    print("Creating visualization 3: Heatmap overlay...")
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Create colored heatmap overlay
    heatmap_colored = plt.cm.hot(heatmap)[:, :, :3]  # RGB
    alpha = 0.5
    overlay = (1 - alpha) * (display_img_gray[:,:,None] / 255.0) + alpha * heatmap_colored
    
    ax.imshow(overlay)
    ax.set_title('Character Heatmap Overlay on Original Image', fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/3_heatmap_overlay.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/3_heatmap_overlay.png")
    plt.close()
    
    # ============================================================================
    # VISUALIZATION 4: Line Segmentation Visualization
    # ============================================================================
    print("Creating visualization 4: Line segmentation...")
    fig, ax = plt.subplots(figsize=(20, 10))
    
    ax.imshow(display_img_gray, cmap='gray')
    
    # Draw horizontal lines at detected line positions
    for i, peak in enumerate(line_peaks):
        ax.axhline(y=peak, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(10, peak, f'Line {i+1}', color='red', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title('Line Segmentation (Red lines indicate detected text lines)', fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/4_line_segmentation.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/4_line_segmentation.png")
    plt.close()
    
    # ============================================================================
    # VISUALIZATION 5: Complete Analysis (Multi-panel)
    # ============================================================================
    print("Creating visualization 5: Complete analysis...")
    fig = plt.figure(figsize=(24, 12))
    
    # Panel 1: Original with bounding boxes
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(display_img_gray, cmap='gray')
    for (x, y, w, h), line_idx in zip(bounding_boxes, lines):
        if line_idx >= 0:
            color = colors[line_idx]
            rect = Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor='none')
            ax1.add_patch(rect)
    ax1.set_title('(a) Original with Character Boxes', fontsize=12)
    ax1.axis('off')
    
    # Panel 2: Heatmap
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(heatmap, cmap='hot')
    ax2.set_title('(b) CRAFT Heatmap', fontsize=12)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # Panel 3: Bounding boxes on heatmap
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(heatmap, cmap='hot')
    for (x, y, w, h) in bounding_boxes:
        rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='cyan', facecolor='none')
        ax3.add_patch(rect)
    ax3.set_title('(c) Detected Character Regions', fontsize=12)
    ax3.axis('off')
    
    # Panel 4: Line segmentation
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(display_img_gray, cmap='gray')
    for i, peak in enumerate(line_peaks):
        ax4.axhline(y=peak, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.set_title('(d) Line Segmentation', fontsize=12)
    ax4.axis('off')
    
    # Panel 5: Vertical projection profile
    ax5 = plt.subplot(2, 3, 5)
    vert_profile = heatmap.sum(axis=1)
    ax5.plot(vert_profile, range(len(vert_profile)), 'b-', linewidth=2)
    ax5.scatter(vert_profile[line_peaks], line_peaks, color='red', s=100, zorder=5)
    ax5.set_title('(e) Vertical Projection Profile', fontsize=12)
    ax5.invert_yaxis()
    ax5.set_xlabel('Intensity Sum')
    ax5.set_ylabel('Y-axis Position')
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    stats_text = f"""
    DETECTION STATISTICS
    ════════════════════════
    
    Total Characters: {len(bounding_boxes)}
    Total Lines: {len(line_peaks)}
    Avg Characters/Line: {len(bounding_boxes)/len(line_peaks):.1f}
    
    Image Size: {heatmap.shape[1]} × {heatmap.shape[0]}
    
    Line Positions (Y):
    {', '.join([str(p) for p in line_peaks])}
    
    Heatmap Range: [{heatmap.min():.3f}, {heatmap.max():.3f}]
    """
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax6.set_title('(f) Detection Statistics', fontsize=12)
    
    plt.suptitle('Sanskrit Manuscript OCR - Complete Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/5_complete_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/5_complete_analysis.png")
    plt.close()
    
    # ============================================================================
    # Save processed data
    # ============================================================================
    print("Saving heatmap data...")
    cv2.imwrite(f'{output_dir}/heatmap_data.png', (heatmap * 255).astype(np.uint8))
    
    # Save bounding box data
    print("Saving bounding box coordinates...")
    with open(f'{output_dir}/bounding_boxes.txt', 'w', encoding='utf-8') as f:
        f.write("Character Bounding Boxes\n")
        f.write("Format: x, y, width, height, line_index\n")
        f.write("="*50 + "\n")
        for i, ((x, y, w, h), line_idx) in enumerate(zip(bounding_boxes, lines)):
            f.write(f"Char_{i+1:04d}: {x:4d}, {y:4d}, {w:4d}, {h:4d}, Line_{line_idx+1 if line_idx>=0 else 'unassigned'}\n")
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nGenerated files in '{output_dir}/':")
    print("  1_bounding_boxes.png      - Character boxes on original image")
    print("  2_character_heatmap.png   - CRAFT model heatmap output")
    print("  3_heatmap_overlay.png     - Heatmap overlaid on original")
    print("  4_line_segmentation.png   - Detected text lines")
    print("  5_complete_analysis.png   - Comprehensive multi-panel view")
    print("  heatmap_data.png          - Raw heatmap data")
    print("  bounding_boxes.txt        - Character coordinates")
    print("\nStatistics:")
    print(f"  - Total characters detected: {len(bounding_boxes)}")
    print(f"  - Total lines detected: {len(line_peaks)}")
    print(f"  - Average characters per line: {len(bounding_boxes)/len(line_peaks):.1f}")
    print("="*80)

if __name__ == "__main__":
    import sys
    
    # Default to sanskrit10.png
    image_path = "sanskrit10.png"
    output_dir = "visualization_output"
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    visualize_all(image_path, output_dir)
