# Mask is Important

About two months ago, when flux fill was just released, we open-sourced a VTON-based virtual try-on model that achieved promising results. In the following period, we conducted numerous experiments and trials. We want to document these training experiences to provide reference for others who might need it in the future.

## Thoughts on Flux Fill

To be honest, flux fill isn't really a mysterious method. Interestingly, even without training, flux fill could already achieve quite good results in most cases. Our training work was more like solving the "last mile" problem. In practice, we found that basically all training could achieve good results with just 5000 steps using batch size=1 and learning rate=1e-5. This made us wonder: how significant was our training work really?

## Comparison between Fine-tuning and LoRA

During our experiments, we found that there were notable differences between fine-tuning and LoRA in the final generated images. Although LoRA could accomplish some virtual try-on tasks, it didn't perform as well as fine-tuning when dealing with complex garments, especially in preserving details like text.

## The Importance of Mask Processing

When training with VTON, we used the pre-processed inpaint mask regions from the VTON dataset and achieved good results. However, when we tried other datasets, problems emerged. Without mature segmentation methods like VTON's, we discovered that mask selection had a huge impact on the final results.

We tried using SAM2 for garment segmentation, but the results weren't ideal. The trained model would stick too closely to the mask shape, leading to a serious problem: long sleeves could only become long sleeves, and short sleeves could only become short sleeves. Worse still, hand-drawn masks would cause severe errors in the final generated images.

After repeated experiments, we realized a key point: the mask preprocessing area needs to be as large as possible. The garment mask shouldn't just frame the garment itself but needs to leave enough drawing space for different garment replacements. However, this brought a new problem: if the redrawing area was too large, it would lead to unstable training.

To solve this problem, we ultimately adopted a combination of OpenPose and SAM2 to redraw human limbs. We paid special attention to ensuring that the drawn mask completely covered the limbs, which was done to counteract the influence of different garment styles. For example, for short-sleeve garments, we needed to ensure their masks showed no trace of whether they were short-sleeve or long-sleeve. Because if the mask itself contained short-sleeve information, the model would tend to generate short sleeves based on this information, affecting try-on accuracy.

This principle also applies to bottom wear processing. For the same model, whether it's long pants, shorts, or skirts, their masks should remain basically consistent. If you can tell from the mask whether it's a skirt or pants, then the final generated garment form would be dominated by the mask rather than determined by the garment itself. This is why masks need to be as general as possible, trying to hide all garment form information.

Of course, this approach needs to be well-balanced. The mask shouldn't be so large that it needs to generate unnecessary information, such as faces or backgrounds. This requires continuous adjustment and balance in practice.

## Importance of Datasets

We experimented with many datasets and found that VTON and DressCode are excellent datasets. If a dataset's accuracy doesn't reach the level of these two datasets, using it for training would be devastating for the model. You'll find that the entire model's precision drops off a cliff, and flux fill's own capability gets severely reduced as well. Therefore, if anyone wants to use specific training sets for training, they must pay attention to the accuracy of data preprocessing.

## Limitations of Current Methods

Although flux fill can solve many virtual try-on problems, its handling of complex patterns hasn't reached our expected level. For example, consider this dress with intricate floral patterns:

![053417_1](https://github.com/user-attachments/assets/a29b7355-f795-4b9f-b7df-04323a55825a)


This type of garment, with its dense, small-scale repeating patterns, is practically a nightmare for all virtual try-on technologies. The complexity and detail of these tiny geometric patterns pose a significant challenge for current methods. Currently, none of the virtual try-on tools in the market can successfully restore these delicate patterns to a highly satisfactory level. The difficulty lies in both preserving the pattern's structure and ensuring its consistent application across the transformed garment. This problem remains a technical challenge that the entire field needs to continue tackling.
