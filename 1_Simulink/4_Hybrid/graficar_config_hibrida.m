% Ploteo config Hibrida

figure(1)
plot(out.tout, out.ruido1, 'DisplayName', 'Se\~{n}al de Ruido en el punto a cancelar')
hold on
grid on
plot(out.tout, out.acc_control,'DisplayName','Se\~{n}al de control')
plot(out.tout, out.error, 'DisplayName', 'Se\~{n}al de error')
legend('Interpreter','Latex', 'FontSize', 12)
xlabel('Tiempo [s]','Interpreter','Latex')
ylabel('Amplitud [u. r.]','Interpreter','Latex')
title('\textbf{Control en Config. Hybrid}','Interpreter','Latex', 'FontSize', 14)
hold off


%Calculo de la atenuación
atenuacion = 20*log( max(out.error)/out.error(length(out.error)) )

%%
% Figura complementaria
RGB = get(groot,"FactoryAxesColorOrder");
H = compose("#%02X%02X%02X",round(RGB*255));

figure(2)
subplot(2,2,1)
plot(out.tout, out.ruido_orig, 'Color', H(1))
title('\textbf{Se\~{n}al de Ruido original}','Interpreter','Latex', 'FontSize', 12)
xlabel('Tiempo [s]','Interpreter','Latex')
ylabel('Amplitud [u. r.]','Interpreter','Latex')
xlim([0 0.02])
grid on

subplot(2,2,2)
plot(out.tout, out.ruido1, 'Color', H(1))
grid on
title('\textbf{Se\~{n}al de Ruido en el punto a cancelar}','Interpreter','Latex','FontSize', 12)
xlabel('Tiempo [s]','Interpreter','Latex')
ylabel('Amplitud [u. r.]','Interpreter','Latex')
xlim([0 0.02])

subplot(2,2,3)
plot(out.tout, out.acc_control,'Color', H(2))
title('\textbf{Se\~{n}al de Acci\''{o}n de control}','Interpreter','Latex','FontSize', 12)
xlabel('Tiempo [s]','Interpreter','Latex')
ylabel('Amplitud [u. r.]','Interpreter','Latex')
grid on
xlim([0 0.1])


subplot(2,2,4)
plot(out.tout, out.error, 'Color', H(3))
title('\textbf{Se\~{n}al de Error}','Interpreter','Latex','FontSize', 12)
xlabel('Tiempo [s]','Interpreter','Latex')
ylabel('Amplitud [u. r.]','Interpreter','Latex')
grid on
