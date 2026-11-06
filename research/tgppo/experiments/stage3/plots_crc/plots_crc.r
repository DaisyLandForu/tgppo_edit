rm(list = ls())
library(dplyr)
library(tidyr)
library(PMCMRplus)
library(PMCMR)
library(openxlsx2)
library(ggplot2)
library(dplyr)
library(ggplot2)
library(scales)
library(patchwork)

rm(list = ls())
setwd(dir = "~/Documents/Intelligentica/Clients/AIMOV/TGPPO-AAAI-26/plots_crc/")

geometric_shifted_mean <- function(x, shift = 100) {
  n <- length(x)
  shifted_values <- x + shift
  gm_shifted <- exp(mean(log(shifted_values))) - shift
  gm_shifted = round(gm_shifted,digits = 2)
  return(gm_shifted)
}

easy_sgm = read_xlsx(file = "finished_instances.xlsx") %>%
  group_by(Instance, Policy) %>%
  summarise(Nnodes = geometric_shifted_mean(Nnodes, shift = 100), .groups = "drop") %>%
  pivot_wider(names_from = Policy, values_from = Nnodes) 

df_easy <- easy_sgm %>%
  transmute(Instance, baseline = tbrant, tgppo) %>%
  mutate(outcome = ifelse(tgppo < baseline, "Good", "Bad"),
         outcome = factor(outcome, levels = c("Good","Bad")))

hard_sgm = read_xlsx(file = "~/Desktop/18OCT2025/13102025//results_tgppo/timelimit_instances.xlsx") %>%
  group_by(Instance, Policy) %>%
  summarise(PDI = geometric_shifted_mean(Integral, shift = 0), .groups = "drop") %>%
  pivot_wider(names_from = Policy, values_from = PDI) 

df_hard <- hard_sgm %>%
  transmute(Instance, baseline = tbrant, tgppo) %>%
  mutate(outcome = ifelse(tgppo < baseline, "Good", "Bad"),
         outcome = factor(outcome, levels = c("Good","Bad")))

df <- bind_rows(
  easy_sgm %>% transmute(Instance, baseline = tbrant, tgppo, Panel = "(a) easy / Nnodes"),
  hard_sgm %>% transmute(Instance, baseline = tbrant, tgppo, Panel = "(b) hard / PDI")
)


# 1. Setup Theme to match AAAI (Times font, 10pt base)
# Note: "serif" usually maps to Times New Roman on most systems.
theme_aaai <- theme_minimal(base_size = 10, base_family = "serif") +
  theme(
    # Ensure text doesn't drop below 7pt (AAAI requirement)
    axis.text = element_text(size = 8, color = "black"),
    axis.title = element_text(size = 10, color = "black"),
    plot.title = element_text(size = 10, face = "bold", hjust = 0.5), # Center title
    panel.grid.minor = element_blank(),
    legend.position = "bottom" # If you want a legend, put it here
  )

p_easy <- ggplot(df_easy, aes(x = baseline, y = tgppo, color = outcome, shape = outcome)) +
  # Add the dashed line first so points sit on top
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  
  # 2. Use SHAPES + High Contrast Colors for Grayscale Safety
  geom_point(alpha = 0.7, size = 2) + 
  
  # Log scales
  scale_x_log10(labels = label_number(scale_cut = cut_short_scale())) +
  scale_y_log10(labels = label_number(scale_cut = cut_short_scale())) +
  coord_equal() +
  
  # FIX: Manual colors and shapes
  # Colors: Black vs Dark Gray (safe for b&w) or Blue vs Orange (colorblind safe)
  # Shapes: 16 (Solid Circle) vs 1 (Hollow Circle) or 4 (Cross)
  scale_color_manual(values = c(Good = "blue", Bad = "red")) + 
  scale_shape_manual(values = c(Good = 16, Bad = 4)) + 
  
  labs(title = "(a) Easy (Nnodes)",
       x = "Baseline (TBranT)", # Shortened for clarity at small sizes
       y = "TGPPO") +
  
  theme_aaai + 
  theme(legend.position = "none") # Hide legend as per your original code

# Print plot to check
print(p_easy)


## Hard plot

p_hard <- ggplot(df_hard, aes(x = baseline, y = tgppo, color = outcome, shape = outcome)) +
  # Add the dashed line first so points sit on top
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  
  # 2. Use SHAPES + High Contrast Colors for Grayscale Safety
  geom_point(alpha = 0.7, size = 2) + 
  
  # Log scales
  scale_x_log10(labels = label_number(scale_cut = cut_short_scale())) +
  scale_y_log10(labels = label_number(scale_cut = cut_short_scale())) +
  coord_equal() +
  
  # FIX: Manual colors and shapes
  # Colors: Black vs Dark Gray (safe for b&w) or Blue vs Orange (colorblind safe)
  # Shapes: 16 (Solid Circle) vs 1 (Hollow Circle) or 4 (Cross)
  scale_color_manual(values = c(Good = "blue", Bad = "red")) + 
  scale_shape_manual(values = c(Good = 16, Bad = 4)) + 
  
  labs(title = "(b) Hard (PDI)",
       x = "Baseline (TBranT)", # Shortened for clarity at small sizes
       y = "TGPPO") +
  
  theme_aaai + 
  theme(legend.position = "none") # Hide legend as per your original code

# Print plot to check
print(p_hard)

# 3. CRITICAL: Save at the exact column width
# AAAI single column width is ~3.3 inches.
# If you save at this size, 10pt font stays 10pt in the PDF.

ggsave("fig_easy_aaai.pdf", plot = p_easy, width = 3.3, height = 3.3, units = "in")
ggsave("fig_hard_aaai.pdf", plot = p_hard, width = 3.3, height = 3.3, units = "in")




###### ONE PLOT FOR BOTH

# --- PLOTTING CODE STARTS HERE ---

# 1. Setup Theme (AAAI: Times font, 10pt base)
theme_aaai <- theme_minimal(base_size = 10, base_family = "serif") +
  theme(
    axis.text = element_text(size = 8, color = "black"),
    axis.title = element_text(size = 10, color = "black"),
    plot.title = element_text(size = 10, face = "bold", hjust = 0.5),
    panel.grid.minor = element_blank(),
    legend.position = "none" # Hide legend inside individual plots
  )

# 2. Easy Plot
p_easy <- ggplot(df_easy, aes(x = baseline, y = tgppo, color = outcome, shape = outcome)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(alpha = 1, size = 2) + 
  scale_x_log10(labels = label_number(scale_cut = cut_short_scale())) +
  scale_y_log10(labels = label_number(scale_cut = cut_short_scale())) +
  coord_equal() +
  # GRAYSCALE SAFE COLORS: Black & Dark Gray
  scale_color_manual(values = c(Good = "black", Bad = "gray30")) + 
  scale_shape_manual(values = c(Good = 16, Bad = 1)) + 
  labs(title = "(a) Easy (Nnodes)", x = "Baseline (TBranT)", y = "TGPPO") +
  theme_aaai

# 3. Hard Plot
p_hard <- ggplot(df_hard, aes(x = baseline, y = tgppo, color = outcome, shape = outcome)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(alpha = 1, size = 2) + 
  scale_x_log10(labels = label_number(scale_cut = cut_short_scale())) +
  scale_y_log10(labels = label_number(scale_cut = cut_short_scale())) +
  coord_equal() +
  # GRAYSCALE SAFE COLORS
  scale_color_manual(values = c(Good = "black", Bad = "gray30")) + 
  scale_shape_manual(values = c(Good = 16, Bad = 1)) + 
  labs(title = "(b) Hard (PDI)", x = "Baseline (TBranT)", y = "TGPPO") +
  theme_aaai

p_easy
p_hard



# 4. COMBINE PLOTS using Patchwork
# The '+' operator places them side-by-side
p_combined <- p_easy + p_hard 

# 5. SAVE
# Width = 7 inches (Standard AAAI double-column width)
# Height = 3.5 inches (Keeps aspect ratio 1:1 for square plots)
ggsave("fig_combined_aaai.pdf", plot = p_combined, width = 7, height = 3.5, units = "in")



# ... [Load libraries and prepare dataframes as before] ...

# 1. Setup "Zero-Margin" Theme
theme_aaai_tight <- theme_minimal(base_size = 9, base_family = "serif") +
  theme(
    axis.text = element_text(size = 8, color = "black"),
    axis.title = element_text(size = 9, color = "black"),
    plot.title = element_text(size = 9, face = "bold", hjust = 0.5),
    panel.grid.minor = element_blank(),
    legend.position = "none",
    # SET MARGINS TO ZERO to remove R's default white border
    plot.margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "pt") 
  )

# 2. Create Plots (Same as before, just applying the tight theme)
p_easy <- ggplot(df_easy, aes(x = baseline, y = tgppo, color = outcome, shape = outcome)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(alpha = 0.7, size = 1.5) + 
  scale_x_log10(labels = label_number(scale_cut = cut_short_scale())) +
  scale_y_log10(labels = label_number(scale_cut = cut_short_scale())) +
  coord_equal() +
  scale_color_manual(values = c(Good = "black", Bad = "gray60")) + 
  scale_shape_manual(values = c(Good = 16, Bad = 4)) + 
  labs(title = "(a) Easy (Nnodes)", x = "Baseline", y = "TGPPO") + # Shortened Axis Title
  theme_aaai_tight

p_hard <- ggplot(df_hard, aes(x = baseline, y = tgppo, color = outcome, shape = outcome)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(alpha = 0.7, size = 1.5) + 
  scale_x_log10(labels = label_number(scale_cut = cut_short_scale())) +
  scale_y_log10(labels = label_number(scale_cut = cut_short_scale())) +
  coord_equal() +
  scale_color_manual(values = c(Good = "black", Bad = "gray60")) + 
  scale_shape_manual(values = c(Good = 16, Bad = 4)) + 
  labs(title = "(b) Hard (PDI)", x = "Baseline", y = "TGPPO") + # Shortened Axis Title
  theme_aaai_tight

p_combined <- p_easy + p_hard 

# 3. SAVE VERY COMPACT
# Height reduced to 2.2 inches. 
# Because margins are 0, this will be very dense.
ggsave("fig_combined_tight.pdf", plot = p_combined, width = 7, height = 2.2, units = "in")


#################.  USING FACET WRAP

library(ggplot2)
library(scales)
library(patchwork) # Required for combining the plots

# 1. Setup Theme
theme_aaai <- theme_minimal(base_size = 10, base_family = "serif") +
  theme(
    axis.text = element_text(size = 8, color = "black"),
    axis.title = element_text(size = 10, color = "black"),
    plot.title = element_text(size = 10, face = "bold", hjust = 0), # Left align title to look like facet label
    panel.grid.minor = element_blank(),
    legend.position = "none",
    plot.margin = margin(10, 10, 10, 10) # Add breathing room
  )

# 2. Create Plot 1: Easy (Filter from combined df)
p_easy <- ggplot(subset(df_combined, type == "(a) Easy (Nnodes)"), 
                 aes(x = baseline, y = tgppo, color = outcome, shape = outcome)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(alpha = 1, size = 2) + 
  scale_x_log10(labels = label_number(scale_cut = cut_short_scale())) +
  scale_y_log10(labels = label_number(scale_cut = cut_short_scale())) +
  coord_equal() + # This is safe now because it's a single plot
  scale_color_manual(values = c(Good = "black", Bad = "gray30")) + 
  scale_shape_manual(values = c(Good = 16, Bad = 1)) + 
  labs(title = "(a) Easy (Nnodes)", x = "Baseline (TBranT)", y = "TGPPO") +
  theme_aaai

# 3. Create Plot 2: Hard (Filter from combined df)
p_hard <- ggplot(subset(df_combined, type == "(b) Hard (PDI)"), 
                 aes(x = baseline, y = tgppo, color = outcome, shape = outcome)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(alpha = 1, size = 2) + 
  scale_x_log10(labels = label_number(scale_cut = cut_short_scale())) +
  scale_y_log10(labels = label_number(scale_cut = cut_short_scale())) +
  coord_equal() + # This is safe now
  scale_color_manual(values = c(Good = "black", Bad = "gray30")) + 
  scale_shape_manual(values = c(Good = 16, Bad = 1)) + 
  labs(title = "(b) Hard (PDI)", x = "Baseline (TBranT)", y = "TGPPO") +
  theme_aaai

# 4. Combine them side-by-side
# The '+' operator comes from the patchwork library
p_combined_final <- p_easy + p_hard
p_combined_final

# 5. Save
ggsave("fig_combined_patchwork.pdf", plot = p_combined_final, width = 7, height = 3.5, units = "in")
