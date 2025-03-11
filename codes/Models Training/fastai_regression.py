from fastai.vision.all import *
import control_utils
import dill


data_path = Path('./path_to data')


def get_label(file_path):
    L_gal, s0, mu = file_path.stem.split('_')
    return [round(float(L_gal), 2), round(float(s0), 2), round(float(mu), 2)]


def mae_custom(inp, targ):
    "Mean absolute error between each variable of `inp` and `targ`."
    f_tot2 = F.mse_loss(*flatten_check(inp, targ))
    f_tot = 0
    f_tot1 = 0
    for inp1, out1 in zip(inp, targ):
        f_tot = f_tot + torch.abs(inp1[0] - out1[0]).mean()+torch.abs(inp1[1] - out1[1]).mean()+torch.abs(inp1[2] - out1[2]).mean() # mab
        f_tot1 = f_tot1 + F.mse_loss(*flatten_check(inp1[0], out1[0]))+F.mse_loss(*flatten_check(inp1[1], out1[1]))+F.mse_loss(*flatten_check(inp1[2], out1[2])) # mse
    f_tot = ((f_tot / len(inp)) + (f_tot1 / len(inp)) + f_tot2) / 3
    return f_tot


def metric_get_shape(inp, targ):
    "The error between each shape of `inp` and `targ`."
    f_tot = 0
    s = np.linspace(start=0, stop=1)
    for inp1, out1 in zip(inp, targ):
        L_gal_in, S0_in, mu_in = unormlized(inp1[0].cpu(), inp1[1].cpu(), inp1[2].cpu())
        L_gal_out, S0_out, mu_out = unormlized(out1[0].cpu(), out1[1].cpu(), out1[2].cpu())
        [x_in, y_in, phi_in] = control_utils.get_shape_hf(0, 0, 0, mu=mu_in, L_gal=L_gal_in, s0=S0_in, s=s)
        [x_out, y_out, phi_out] = control_utils.get_shape_hf(0, 0, 0, mu=mu_out, L_gal=L_gal_out, s0=S0_out, s=s)
        temp = F.mse_loss(x_in, x_out) + F.mse_loss(y_in, y_out)
        if temp.isnan():
            continue
        else:
            f_tot = f_tot + temp
    f_tot = f_tot / len(inp)
    return f_tot.cuda()


def mse_custom(inp, targ):
    "The error between each shape of `inp` and `targ`."
    f_tot = 0
    for inp1, out1 in zip(inp, targ):
        Ninp = tensor(unormlized(inp1[0], inp1[1], inp1[2]))
        Nout = tensor(unormlized(out1[0], out1[1], out1[2]))
        temp = F.mse_loss( Ninp, Nout)
        f_tot = f_tot + temp
    f_tot = f_tot / len(inp)
    return f_tot

def unormlized(L_gal, S0, mu):
    mu_min = -0.462
    mu_max = 1
    L_gal_max = 4
    L_gal_min = 1
    S0_max = L_gal_max
    S0_min = 0
    not_norm_L_gal = (L_gal * (L_gal_max - L_gal_min) + L_gal_min)
    not_norm_S0 = (S0 * (S0_max - S0_min) + S0_min)
    not_norm_mu = (mu * (mu_max - mu_min) + mu_min)
    return not_norm_L_gal, not_norm_S0, not_norm_mu

if __name__ == '__main__':

    ds = DataBlock(blocks=(ImageBlock, RegressionBlock(n_out=3)),  get_items=get_image_files, splitter=RandomSplitter(),
                get_y=get_label, item_tfms=Resize(460), batch_tfms=[])

    dls = ds.dataloaders(data_path, bs=128)

    dls.show_batch()


    model = vision_learner(dls, resnet50 ,loss_func=nn.HuberLoss(delta=0.1) ,metrics=[mse_custom, metric_get_shape])

    model.to_fp16()

    plt.figure('2')

    lr_min, lr_steep, lr_valley, lr_slide = model.lr_find(suggest_funcs=(minimum, steep, valley, slide))

    plt.show()

    model.fine_tune(50, lr_valley) # cbs=[TrainEvalCallback(), Recorder]
    model.save('model_2Delstica_HuberLoss_resnet50')
    model.export('./models/export_models/model_2Delstica_HuberLoss_resnet50',pickle_module=dill)

    plt.figure('3')
    model.show_results()

    interp = SegmentationInterpretation.from_learner(model)

    losses,idxs = interp.top_losses()

    plt.show()

    # Test the model

    test_folder = data_path/'Test_folder'
    test_items = list(test_folder.glob("*"))
    test_dl = dls.test_dl(test_items)

    import random
    import PIL as Image

    for image_path in random.sample(test_dl.items, 15):
        print(image_path)
        labels = get_label(image_path)
        unnorm_label = unormlized(*labels)


        pred = model.predict(image_path)[0]

        unnorm_pred = unormlized(*pred)

        # Mean Squared Error 
        MSE = np.square(np.subtract(labels,pred)).mean() 
        Unnorm_MSE = np.square(np.subtract(unnorm_label,unnorm_pred)).mean() 

        print("Label:", labels)
        print("Pred:", pred)
        print("MSE:", MSE)


        print("UNorm Label:", unnorm_label)
        print("Unnorm pred:", unnorm_pred)
        print("Unnorm MSE:", Unnorm_MSE)

        plt.imshow(Image.open(image_path))
        plt.axis('off')
        plt.show()
        shape = control_utils.get_shape_hf(0,0,0,L_gal=unnorm_pred[0],s0=unnorm_pred[1],mu=unnorm_pred[2],s=np.linspace(0,1))
        true_shape = control_utils.get_shape_hf(0,0,0,L_gal=unnorm_label[0],s0=unnorm_label[1],mu=unnorm_label[2],s=np.linspace(0,1))
        plt.plot(shape[0],shape[1],'-k', label= 'pred')
        plt.plot(true_shape[0],true_shape[1], label = 'true')
        plt.legend()
        plt.title('prediction')
        plt.axis('equal')
        plt.axis('off')
        plt.show()




