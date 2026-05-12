import os

import uproot
import numpy as np
import tensorflow as tf

from configs import common, layer_configs, layer_filter_rules
from models import architectures, losses
import plotting


class Trainer:
    def __init__(self, layer, axis):
        if axis not in {"x", "y"}:
            raise ValueError(f"Invalid axis '{axis}'. Valid values are 'x' or 'y'.")

        self.layer = layer
        self.axis = axis

        if layer not in layer_configs:
            available = ", ".join(sorted(layer_configs.keys())) or "<none>"
            raise KeyError(f"Unknown layer '{layer}'. Available layers: {available}")

        layer_config = layer_configs[layer]
        self.config = {**common, **layer_config}

        self.batch_size = self.config.get("batch_size")
        self.epochs = self.config.get("epochs")
        self.loss_name = self.config.get("loss_name")
        self.modelname = self.config.get("modelname", "baseline_model")
        self.pitch = self.config.get(f"pitch_{axis}")
        self.checkpoint = self.config.get(f"checkpoint_{axis}")
        self.model_dest = self.config.get(f"model_dest_{axis}", f"checkpoints/{self.layer}_{self.axis}.keras")
        self.debug_predictions = bool(self.config.get("debug_predictions", False))
        self.input_file = self.config.get("input_file")
        self.output_scale = self.config.get("output_scale")
        self.first_guess = self.config.get("first_guess")
        self.input_prepared = False
        self.model = None

    def _ensure_model_loaded(self):
        """Lazily load the model if it hasn't been trained or loaded yet.

        Priority:
          1. Already in memory (self.model is not None) — nothing to do.
          2. Full saved Keras model at self.model_dest.
          3. Build architecture + load checkpoint weights from self.checkpoint.
          4. Raise a clear error if neither is available.
        """
        if self.model is not None:
            return

        loss_func = getattr(losses, self.loss_name)
        custom_objects = {
            self.loss_name: loss_func,
            "mse_position": losses.mse_position,
        }

        if os.path.exists(self.model_dest):
            print(f"Loading full model from: {self.model_dest}")
            self.model = tf.keras.models.load_model(self.model_dest, custom_objects=custom_objects)
            return

        if self.checkpoint and os.path.exists(self.checkpoint):
            print(f"Full model not found. Building architecture and loading weights from: {self.checkpoint}")
            if not self.input_prepared:
                raise RuntimeError(
                    "Cannot rebuild model from checkpoint without prepared input "
                    "(need input shape). Call prepare_train_test_input() first."
                )
            x = self.cluster_x_test if self.axis == "x" else self.cluster_y_test
            input_len = x.shape[1]
            input_dim = 16 if self.axis == "x" else 24
            inputs = tf.keras.layers.Input(shape=(input_len, 1), name=f"pixel_projection_{self.axis}")
            angles = tf.keras.layers.Input(shape=(2,), name="angles")
            charges = tf.keras.layers.Input(shape=(1,), name="cluster_charge")
            model_fn = getattr(architectures, self.modelname)
            model = model_fn(inputs, angles, charges, input_dim)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss=loss_func,
                metrics=[loss_func, losses.mse_position],
            )
            model.load_weights(self.checkpoint)
            self.model = model
            return

        raise RuntimeError(
            f"No trained model found. Expected either:\n"
            f"  full model : {self.model_dest}\n"
            f"  checkpoint : {self.checkpoint}\n"
            "Call train() first, or place the model/checkpoint files in the expected paths."
        )

    def _first_simhit_per_event(self, values, branch_name):
        first_values = np.empty(len(values), dtype=np.float32)
        for idx, event_values in enumerate(values):
            arr = np.asarray(event_values)
            flat = arr.reshape(-1)
            if flat.size == 0:
                raise ValueError(f"{branch_name}[{idx}] has no SimHit entries.")
            first_values[idx] = np.float32(flat[0])
        return first_values.reshape(-1, 1)

    def prepare_train_test_input(self, train_split=0.9, seed=42):
        if not self.input_file:
            raise ValueError("Missing 'input_file' in configuration.")
        if self.layer not in layer_filter_rules:
            raise KeyError(f"No filter rule defined for layer '{self.layer}'.")

        print(f"Opening ROOT file: {self.input_file}")
        with uproot.open(self.input_file) as root_file:
            tree = root_file["Events"]

            needed_branches = [
                "Layer", "Ladder", "Module",
                "CotAlpha", "CotBeta",
                "ClusterCenter_x", "ClusterCenter_y",
                "Cluster_charge",
                "Cluster", "Cluster_x", "Cluster_y",
                "SimHit_x", "SimHit_y", "Generic_x", "Generic_y"
            ]
            print(f"Reading {len(needed_branches)} branches from {tree.num_entries} entries...")
            data = tree.arrays(needed_branches, library="np")

        # --- Layer / module filter ---
        filter_fn = layer_filter_rules[self.layer]
        mask = filter_fn(data["Layer"], data["Ladder"], data["Module"])
        n_total = mask.sum()
        print(f"Clusters passing '{self.layer}' filter: {n_total} / {len(mask)}")

        def apply(key):
            return data[key][mask]

        cot_alpha     = apply("CotAlpha")
        cot_beta      = apply("CotBeta")
        cluster_2d    = apply("Cluster")       # 2-D pixel array (jagged or fixed shape)
        cluster_x     = apply("Cluster_x")     # x projection (1-D strip)
        cluster_y     = apply("Cluster_y")     # y projection (1-D strip)
        center_x      = apply("ClusterCenter_x") * self.output_scale 
        center_y      = apply("ClusterCenter_y") * self.output_scale 
        cluster_charge = apply("Cluster_charge")
        simhit_x      = apply("SimHit_x") * self.output_scale 
        simhit_y      = apply("SimHit_y") * self.output_scale 
        generic_x      = apply("Generic_x") * self.output_scale 
        generic_y      = apply("Generic_y") * self.output_scale 

        print("\n--- Sanity check: Print properties of first 5 clusters ---")
        n_check = min(5, n_total)
        np.set_printoptions(linewidth=300, suppress=True)

        # Formatter to print 1-D arrays in a compact form
        def fmt_1d(arr):
            values = np.asarray(arr).reshape(-1)
            return "[" + " ".join(f"{float(value):.2f}" for value in values) + "]"

        for i in range(n_check):
            c2d  = np.array(cluster_2d[i])
            cx   = np.array(cluster_x[i])
            cy   = np.array(cluster_y[i])

            print(f"  Cluster {i}: {cx.shape},{cy.shape}")
            print(f"    Cluster_x : {fmt_1d(cx)}")
            print(f"    Cluster_y : {fmt_1d(cy)}")
            print(f"    Charge    : {float(np.asarray(cluster_charge[i]).reshape(-1)[0]):.2f}")
            print(f"    CotAlpha  : {cot_alpha[i]:.2f}")
            print(f"    CotBeta   : {cot_beta[i]:.2f}")

        # --- Train / test split ---
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
        perm = np.random.permutation(n_total)
        n_train = int(n_total * train_split)

        def split(arr):
            arr = np.array(arr) if not isinstance(arr, np.ndarray) else arr
            return arr[perm[:n_train]], arr[perm[n_train:]]

        self.cot_alpha_train,      self.cot_alpha_test      = split(cot_alpha)
        self.cot_beta_train,       self.cot_beta_test       = split(cot_beta)
        self.cluster_x_train,      self.cluster_x_test      = split(cluster_x)
        self.cluster_y_train,      self.cluster_y_test      = split(cluster_y)
        self.center_x_train,       self.center_x_test       = split(center_x)
        self.center_y_train,       self.center_y_test       = split(center_y)
        self.cluster_charge_train, self.cluster_charge_test = split(cluster_charge)
        self.simhit_x_train,       self.simhit_x_test       = split(simhit_x)
        self.simhit_y_train,       self.simhit_y_test       = split(simhit_y)
        self.generic_x_train,       self.generic_x_test       = split(generic_x)
        self.generic_y_train,       self.generic_y_test       = split(generic_y)

        print(f"Train / test split: {n_train} / {n_total - n_train} clusters")
        self.input_prepared = True


    def train(self, validation_split=0.02):
        if not self.input_prepared:
            self.prepare_train_test_input()

        x_train = self.cluster_x_train if self.axis == "x" else self.cluster_y_train
        x_test = self.cluster_x_test if self.axis == "x" else self.cluster_y_test

        angles_train = np.column_stack([self.cot_alpha_train, self.cot_beta_train]).astype(np.float32)
        angles_test = np.column_stack([self.cot_alpha_test, self.cot_beta_test]).astype(np.float32)

        charge_train = np.asarray(self.cluster_charge_train, dtype=np.float32).reshape(-1, 1)
        charge_test = np.asarray(self.cluster_charge_test, dtype=np.float32).reshape(-1, 1)

        center_train = np.asarray(self.center_x_train if self.axis == "x" else self.center_y_train, dtype=np.float32).reshape(-1, 1)
        center_test = np.asarray(self.center_x_test if self.axis == "x" else self.center_y_test, dtype=np.float32).reshape(-1, 1)
        generic_train = np.asarray(self.generic_x_train if self.axis == "x" else self.generic_y_train, dtype=np.float32).reshape(-1, 1)
        generic_test = np.asarray(self.generic_x_test if self.axis == "x" else self.generic_y_test, dtype=np.float32).reshape(-1, 1)
        if self.axis == "x":
            simhit_train = self._first_simhit_per_event(self.simhit_x_train, "SimHit_x_train")
            simhit_test = self._first_simhit_per_event(self.simhit_x_test, "SimHit_x_test")
        else:
            simhit_train = self._first_simhit_per_event(self.simhit_y_train, "SimHit_y_train")
            simhit_test = self._first_simhit_per_event(self.simhit_y_test, "SimHit_y_test")

        # Final hit position is cluster center + NN prediction, so we train the NN to predict the offset from the cluster center to the SimHit position.
        if self.first_guess == "center":
            target_offset_train = simhit_train - center_train
            target_offset_test = simhit_test - center_test
        elif self.first_guess == "generic":
            target_offset_train = simhit_train - generic_train
            target_offset_test = simhit_test - generic_test
        print(np.mean(target_offset_test), np.std(target_offset_test))
        input_len = x_train.shape[1]

        #13(21) pixels in x(y) + 2 angles + 1 charge = 16(24) input features for x(y) projection
        input_dim = 16 if self.axis == "x" else 24

        inputs = tf.keras.layers.Input(shape=(input_len, 1), name=f"pixel_projection_{self.axis}")
        angles = tf.keras.layers.Input(shape=(2,), name="angles")
        charges = tf.keras.layers.Input(shape=(1,), name="cluster_charge")
        model_fn = getattr(architectures, self.modelname)
        model = model_fn(inputs, angles, charges, input_dim)
        loss_func = getattr(losses, self.loss_name)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=loss_func,
            metrics=[loss_func, losses.mse_position],
            run_eagerly=False,
        )

        callbacks = []
        if self.checkpoint:
            checkpoint_dir = os.path.dirname(self.checkpoint)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
            if os.path.exists(self.checkpoint):
                print(f"Loading weights from checkpoint: {self.checkpoint}")
                model.load_weights(self.checkpoint)
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.checkpoint,
                    save_best_only=True,
                    save_weights_only=True,
                    monitor=self.loss_name,
                    save_freq="epoch",
                )
            )
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )
        callbacks.append(early_stop)

        history = model.fit(
            [x_train[:, :, np.newaxis], angles_train, charge_train],
            target_offset_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            validation_split=validation_split,
            verbose=1,
        )

        if self.checkpoint:
            print(f"Saving final weights to checkpoint: {self.checkpoint}")
            model.save_weights(self.checkpoint)

        model_dir = os.path.dirname(self.model_dest)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        print(f"Saving full model to: {self.model_dest}")
        model.save(self.model_dest)

        pred_train = model.predict([x_train[:, :, np.newaxis], angles_train, charge_train], batch_size=self.batch_size, verbose=0)
        pred_test = model.predict([x_test[:, :, np.newaxis], angles_test, charge_test], batch_size=self.batch_size, verbose=0)

        self.train_offset_pred = pred_train[:, 0:1]
        self.test_offset_pred = pred_test[:, 0:1]
        self.train_uncertainty_pred = pred_train[:, 1:2]
        self.test_uncertainty_pred = pred_test[:, 1:2]

        if self.first_guess == "center":
            self.train_simhit_pred = center_train + self.train_offset_pred
            self.test_simhit_pred = center_test + self.test_offset_pred
        elif self.first_guess == "generic":
            self.train_simhit_pred = generic_train + self.train_offset_pred
            self.test_simhit_pred = generic_test + self.test_offset_pred
        self.model = model
        self.history = history
        self.train_target_offset = target_offset_train
        self.test_target_offset = target_offset_test

        os.makedirs("plots", exist_ok=True)
        history_plot = f"plots/{self.layer}_{self.axis}_history.png"
        plotting.plot_dnn_loss(history.history, history_plot)
        plotting.plot_nll_and_mse(history.history, f"plots/{self.layer}_{self.axis}")

        print(f"Training done: {len(x_train)} train / {len(x_test)} test clusters")

    def test(self):
        if not self.input_prepared:
            self.prepare_train_test_input()
        self._ensure_model_loaded()

        x_test = self.cluster_x_test if self.axis == "x" else self.cluster_y_test
        angles_test = np.column_stack([self.cot_alpha_test, self.cot_beta_test]).astype(np.float32)
        charge_test = np.asarray(self.cluster_charge_test, dtype=np.float32).reshape(-1, 1)

        center_test = np.asarray(self.center_x_test if self.axis == "x" else self.center_y_test, dtype=np.float32).reshape(-1, 1)
        generic_test = np.asarray(self.generic_x_test if self.axis == "x" else self.generic_y_test, dtype=np.float32).reshape(-1, 1)
        if self.axis == "x":
            simhit_test = self._first_simhit_per_event(self.simhit_x_test, "SimHit_x_test")
        else:
            simhit_test = self._first_simhit_per_event(self.simhit_y_test, "SimHit_y_test")

        if self.first_guess == "center":
            target_offset_test = simhit_test - center_test
        elif self.first_guess == "generic":
            target_offset_test = simhit_test - generic_test
        pred = self.model.predict([x_test[:, :, np.newaxis], angles_test, charge_test], batch_size=self.batch_size, verbose=0)
        self.pred = pred

        residuals_native = pred[:, 0] - target_offset_test[:, 0] # native == in cm
        uncertainty_native = pred[:, 1]
        conversion_to_microns = 1e4 / self.output_scale

        residuals = residuals_native * conversion_to_microns
        uncertainties = uncertainty_native * conversion_to_microns
        if self.debug_predictions:
            print("First five predictions (in um):")
            for i in range(min(5, len(pred))):
                print(f"  Cluster {i}: Prediction = {pred[i, 0]*conversion_to_microns:.2f} um, Target = {target_offset_test[i, 0]*conversion_to_microns:.2f} um, Residual = {residuals[i]:.2f} um, Uncertainty = {uncertainties[i]:.2f} um")

        pulls = residuals / np.maximum(uncertainties, 1e-6)

        self.resolution = float(np.std(residuals))
        self.bias = float(np.mean(residuals))

        print("Test in offset frame (SimHit - ClusterCenter)")
        print("Residuals mean and std (microns): {:.3f} +/- {:.3f}".format(self.bias, self.resolution))
        print("Pulls mean and std: {:.3f} +/- {:.3f}".format(float(np.mean(pulls)), float(np.std(pulls))))

        os.makedirs("plots", exist_ok=True)
        residuals_output_file = f"plots/NNwith{self.first_guess}_{self.layer}_{self.axis}_residuals.pdf"
        pulls_output_file = f"plots/NNwith{self.first_guess}_{self.layer}_{self.axis}_pulls.pdf"
        plot_name = f"NNwith{self.first_guess}_{self.layer}_{self.axis}"
        plotting.plot_residuals(residuals, residuals_output_file, plot_type="Residuals", name=plot_name)
        plotting.plot_residuals(pulls, pulls_output_file, plot_type="Pulls", name=plot_name)
        plotting.plot_uncertainties(uncertainties, f"plots/NNwith{self.first_guess}_{self.layer}_{self.axis}_uncertainties.pdf")

    def visualize(self, n_to_plot=10):
        if not self.input_prepared:
            self.prepare_train_test_input()

        self._ensure_model_loaded()

        if not hasattr(self, "pred"):
            self.test()

        x_test = self.cluster_x_test if self.axis == "x" else self.cluster_y_test
        center_test = np.asarray(
            self.center_x_test if self.axis == "x" else self.center_y_test,
            dtype=np.float32,
        ).reshape(-1, 1)
        generic_test = np.asarray(
            self.generic_x_test if self.axis == "x" else self.generic_y_test,
            dtype=np.float32,
        ).reshape(-1, 1)
        if self.axis == "x":
            simhit_test = self._first_simhit_per_event(self.simhit_x_test, "SimHit_x_test")
        else:
            simhit_test = self._first_simhit_per_event(self.simhit_y_test, "SimHit_y_test")
        if self.first_guess == "center":
            target_offset_test = (simhit_test - center_test) * 1e4 / self.output_scale 
        elif self.first_guess == "generic":
            target_offset_test = (simhit_test - generic_test) * 1e4 / self.output_scale
        pred_offset_test = self.pred[:, 0] * 1e4 / self.output_scale
        pred_uncertainty_test = self.pred[:, 1] * 1e4 / self.output_scale

        n_to_plot = min(n_to_plot, len(x_test))
        plotting_data_sets = []
        plotting_file_name = f"plots/NNwith{self.first_guess}_{self.layer}_{self.axis}.pdf"

        for idx in range(n_to_plot):
            data_set = {
                "cluster": x_test[idx],
                "angles": np.array([self.cot_alpha_test[idx], self.cot_beta_test[idx]], dtype=np.float32),
                "prediction_uncertainty": np.array([pred_offset_test[idx], pred_uncertainty_test[idx]], dtype=np.float32),
                "position": float(target_offset_test[idx, 0]),
                "pixel_pitch": self.pitch,
                "resolution": getattr(self, "resolution", 0.0),
                "bias": getattr(self, "bias", 0.0),
            }
            plotting_data_sets.append(data_set)

        os.makedirs("plots", exist_ok=True)
        plotting.plot_clusters(plotting_data_sets, plotting_file_name)

    def plot_otherMethods(self):
        simhit_y = self._first_simhit_per_event(self.simhit_y_test, "SimHit_y_train")[:,0]
        residuals_center_y = ( self.center_y_test - simhit_y ) * 1e4 / self.output_scale 
        residuals_generic_y = ( self.generic_y_test - simhit_y ) * 1e4 / self.output_scale
        plotting.plot_residuals(residuals_center_y, f"plots/center_{self.layer}_y_residuals.pdf" , plot_type="Residuals", name=f"center_y")
        plotting.plot_residuals(residuals_generic_y, f"plots/generic_{self.layer}_y_residuals.pdf", plot_type="Residuals", name=f"generic_y")
        
        simhit_x = self._first_simhit_per_event(self.simhit_x_test, "SimHit_x_train")[:,0]
        residuals_center_x = ( self.center_x_test - simhit_x ) * 1e4 / self.output_scale
        residuals_generic_x = ( self.generic_x_test - simhit_x ) * 1e4 / self.output_scale
        plotting.plot_residuals(residuals_center_x, f"plots/center_{self.layer}_x_residuals.pdf" , plot_type="Residuals", name=f"center_x")
        plotting.plot_residuals(residuals_generic_x, f"plots/generic_{self.layer}_x_residuals.pdf", plot_type="Residuals", name=f"generic_x")


if __name__ == "__main__":
    trainer = Trainer(layer="L1U", axis="y")
    trainer.prepare_train_test_input()
    trainer.train()
    trainer.test()
    trainer.plot_otherMethods()
    trainer.visualize()
